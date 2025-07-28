import dataclasses
import json
import logging
import logging.config
import logging.handlers
import multiprocessing
import multiprocessing.queues
import pathlib
import pickle
import queue
import threading
import time
import traceback
import typing
import warnings
import zipfile

import numpy
import tqdm

from .config import Config
from .flow import FlowSource, Direction
from .bitmap import BitmapSource
from .accumulator import Accumulator
from .output import VideoOutput, ZipOutput, NumpyOutput
from .output.render import render1d, render2d
from .utils import multiply_arrays, binarize_arrays, absmax, upscale_array


def setup_logging(log_queue: multiprocessing.Queue, level: str):
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(handler)


def logging_listener_process(log_queue: multiprocessing.Queue, config: dict):
    logging.config.dictConfig(config)
    logger = logging.getLogger()
    listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    listener.start()
    while True:
        try:
            record = log_queue.get()
            if record is None:
                break
            logger.handle(record)
        except KeyboardInterrupt:
            pass
    listener.stop()


class SourceProcess(multiprocessing.Process):

    def __init__(self,
            source: FlowSource.Builder | BitmapSource,
            out_queue: multiprocessing.Queue,
            metadata_queue: multiprocessing.Queue,
            log_queue: multiprocessing.Queue,
            log_level: str):
        multiprocessing.Process.__init__(self)
        self.source = source
        self.queue = out_queue
        self.shape_queue = metadata_queue
        self.log_queue = log_queue
        self.log_level = log_level

    def run(self):
        setup_logging(self.log_queue, self.log_level)
        logger = logging.getLogger(__name__)
        put_none = True
        try:
            with self.source as source:
                logger.debug("Entered source %s", source.__class__.__name__)
                self.shape_queue.put((
                    source.width,
                    source.height,
                    source.framerate,
                    source.length,
                    source.direction if isinstance(source, FlowSource) else None
                ))
                try:
                    for item in source:
                        self.queue.put(item)
                except KeyboardInterrupt:
                    logger.debug("Source process '%s' got interrupted", source.__class__.__name__)
                    put_none = False
                except Exception as err:
                    logger.error("Source process '%s' encountered an error: %s", source.__class__.__name__, err)
                    put_none = False
                    traceback.print_exc()
        except Exception as err:
            logger.error("Source process '%s', encountered an error: %s", self.source, err)
            put_none = False
            traceback.print_exc()
        if put_none:
            self.queue.put(None)
        logger.debug("End of source process '%s'", self.source)
        logging.shutdown()


class OutputProcess(multiprocessing.Process):

    def __init__(self,
            output: VideoOutput,
            in_queue: multiprocessing.Queue,
            log_queue: multiprocessing.Queue,
            log_level: str):
        multiprocessing.Process.__init__(self)
        self.output = output
        self.queue = in_queue
        self.log_queue = log_queue
        self.log_level = log_level

    def run(self):
        setup_logging(self.log_queue, self.log_level)
        logger = logging.getLogger(__name__)
        logger.debug("Starting output process '%s'", self.output.__class__.__name__)
        with self.output:
            while True:
                try:
                    frame = self.queue.get()
                    if frame is None:
                        break
                    self.output.feed(frame)
                except KeyboardInterrupt:
                    logger.debug("Output process got interrupted")
                    break # was 'continue' before, but why?
                except Exception as err:
                    logger.error("Output process encountered an exception: %s", err)
                    traceback.print_exc()
                    break
        logger.debug("End of output process '%s'", self.output.__class__.__name__)
        logging.shutdown()


class Pipeline:

    @dataclasses.dataclass
    class Status:

        cursor: int
        total: int | None
        elapsed: float
        error: str | None

    FLOW_MERGING_FUNCTIONS: dict[str, typing.Callable[[list[numpy.ndarray]], numpy.ndarray]] = {
        "first": lambda flows: flows[0],
        "sum": lambda flows: numpy.sum(flows, axis=0),
        "average": lambda flows: numpy.sum(flows, axis=0) / len(flows),
        "difference": lambda flows: flows[0] - sum(flows[1:]),
        "product": multiply_arrays,
        "maskbin": lambda flows: multiply_arrays([flows[0]] + binarize_arrays(flows[1:])),
        "masklin": lambda flows: multiply_arrays([flows[0]] + [numpy.abs(f) for f in flows[1:]]),
        "absmax": absmax,
    }

    def __init__(self,
            cfg: Config,
            log_level: str = "DEBUG",
            log_handler: str = "null",
            log_path: pathlib.Path = pathlib.Path("transflow.log"),
            round_flow: bool = False,
            export_flow: bool = False,
            execute: bool = False,
            replace: bool = False,
            checkpoint_every: int | None = None,
            checkpoint_end: bool = False,
            export_config: bool = True,
            safe: bool = True,
            preview_output: bool = False,
            cancel_event: threading.Event | None = None,
            status_queue: multiprocessing.queues.Queue | None = None):
        self.config = cfg
        self.log_level = log_level
        self.log_handler = log_handler
        self.log_path = log_path
        self.round_flow = round_flow
        self.export_flow = export_flow
        self.execute = execute
        self.replace = replace
        self.checkpoint_every = checkpoint_every
        self.checkpoint_end = checkpoint_end or safe
        self.export_config = export_config or safe
        self.safe = safe
        self.preview_output = preview_output
        self.cancel_event = cancel_event
        self.status_queue = status_queue
        self.log_queue = multiprocessing.Queue()
        self.metadata_queue: multiprocessing.Queue | None = None
        self.flow_queue: multiprocessing.Queue | None = None
        self.bitmap_queue: multiprocessing.Queue | None = None
        self.output_queues: list[multiprocessing.Queue] = []
        self.log_listener: multiprocessing.Process | None = None
        self.flow_process: SourceProcess | None = None
        self.flow_source: FlowSource.Builder | None = None
        self.bitmap_process: SourceProcess | None = None
        self.output_processes: list[OutputProcess] = []
        self.extra_flow_sources: list[FlowSource.Builder] = []
        self.extra_flow_queues: list[multiprocessing.Queue] = []
        self.extra_flow_processes: list[SourceProcess] = []
        self.merge_flows = self.FLOW_MERGING_FUNCTIONS[cfg.flows_merging_function]
        self.flow_output: NumpyOutput | None = None
        self.accumulator: Accumulator | None = None
        self.ckpt_meta: dict = {}
        self.fs_width: int | None = None
        self.fs_height: int | None = None
        self.fs_framerate: float | None = None
        self.fs_length: int | None = None
        self.fs_direction: Direction | None = None
        self.bs_framerate: float | None = None
        self.bs_length: int | None = None
        self.fs_width_factor: int = 1
        self.fs_height_factor: int = 1
        self.cursor: int = 0

    @property
    def has_output(self) -> bool:
        return self.config.bitmap_path is not None\
            or self.config.output_intensity\
            or self.config.output_heatmap\
            or self.config.output_accumulator

    def export_checkpoint(self):
        output = ZipOutput(self.config.get_secondary_output_path(f"_{self.cursor:05d}.ckpt.zip"), self.replace)
        output.write_meta({
            "config": self.config.todict(),
            "cursor": self.cursor,
            "framerate": self.fs_framerate,
            "timestamp": time.time(),
        })
        output.write_object("accumulator.bin", self.accumulator)
        output.close()
        self.logger.debug("Exported checkpoint at cursor %d", self.cursor)

    @property
    def expected_length(self) -> int | None:
        if self.fs_length is not None and self.bs_length is not None:
            return min(self.fs_length, self.bs_length)
        if self.fs_length is not None:
            return self.fs_length
        if self.bs_length is not None:
            return self.bs_length
        return None

    def _setup_logging(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers = [h.strip() for h in self.log_handler.split(",")]
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s] %(levelname)s %(name)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {},
            "root": {
                "handlers": log_handlers,
            }
        }
        if "file" in log_handlers:
            logging_config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "filename": self.log_path.as_posix(),
                "formatter": "standard",
            }
        if "stream" in log_handlers:
            logging_config["handlers"]["stream"] = {
                "class": "logging.StreamHandler",
                "formatter": "standard"
            }
        if "null" in log_handlers:
            logging_config["handlers"]["null"] = {
                "class": "logging.NullHandler"
            }
        self.log_listener = multiprocessing.Process(target=logging_listener_process, args=(self.log_queue, logging_config))
        self.log_listener.start()
        setup_logging(self.log_queue, self.log_level)
        # TODO: close log process nicely if an error occurs while initializing the pipeline process

    def _setup_checkpoint(self):
        self.ckpt_meta = {}
        if not self.config.flow_path.endswith(".ckpt.zip"):
            return
        with zipfile.ZipFile(self.config.flow_path) as archive:
            with archive.open("meta.json") as file:
                self.ckpt_meta = json.loads(file.read().decode())
            with archive.open("accumulator.bin") as file:
                self.accumulator = pickle.load(file)
        self.config = Config.fromdict(self.ckpt_meta["config"])
        self.logger.debug("Flow path is a checkpoint, restarting from frame %d.", self.ckpt_meta["cursor"])
        self.config.seek_time += self.ckpt_meta["cursor"] / self.ckpt_meta["framerate"]
        if self.config.duration_time is not None:
            self.config.duration_time -= self.ckpt_meta["cursor"] / self.ckpt_meta["framerate"]
        # TODO: allow overriding checkpoint config
        # self.config.flow_path = self.ckpt_meta["config"]["flow_path"]
        # self.config.seed = self.ckpt_meta["config"]["seed"]

    def _setup_flow_sources(self):
        assert self.metadata_queue is not None
        fs_args = {
            "use_mvs": self.config.use_mvs,
            "mask_path": self.config.mask_path,
            "kernel_path": self.config.kernel_path,
            "cv_config": self.config.cv_config,
            "flow_filters": self.config.flow_filters,
            "size": self.config.size,
            "direction": self.config.direction,
            "seek_ckpt": self.ckpt_meta.get("cursor"),
            "seek_time": self.config.seek_time,
            "duration_time": self.config.duration_time,
            "repeat": self.config.repeat,
            "lock_expr": self.config.lock_expr,
            "lock_mode": self.config.lock_mode
        }
        self.flow_source = FlowSource.from_args(self.config.flow_path, **fs_args)
        self.flow_queue = multiprocessing.Queue(maxsize=1)
        self.flow_process = SourceProcess(self.flow_source, self.flow_queue, self.metadata_queue, self.log_queue, self.log_level)
        self.flow_process.start()
        self.logger.debug("Started flow process")
        for i, extra_flow_path in enumerate(self.config.extra_flow_paths):
            self.extra_flow_sources.append(FlowSource.from_args(extra_flow_path, **fs_args))
            self.extra_flow_queues.append(multiprocessing.Queue(maxsize=1))
            self.extra_flow_processes.append(SourceProcess(self.extra_flow_sources[-1], self.extra_flow_queues[-1], self.metadata_queue, self.log_queue, self.log_level))
            self.extra_flow_processes[-1].start()
            self.logger.debug("Started extra flow process no. %d", i)

    def _wait_for_flow_sources(self):
        assert self.metadata_queue is not None and self.flow_process is not None
        flow_sources_to_load = 1 + len(self.extra_flow_processes)
        flow_sources_loaded = 0
        while True:
            try:
                shape_info = self.metadata_queue.get(timeout=1)
                if flow_sources_loaded == 0:
                    (self.fs_width, self.fs_height, self.fs_framerate, self.fs_length, self.fs_direction) = shape_info
                flow_sources_loaded += 1
                self.logger.debug("Received metadata message from a flow process [%d/%d]", flow_sources_loaded, flow_sources_to_load)
                if flow_sources_loaded >= flow_sources_to_load:
                    break
            except queue.Empty as exc:
                if self.flow_process.is_alive() and all(p.is_alive() for p in self.extra_flow_processes):
                    continue
                raise RuntimeError("Flow process died during initialization.") from exc
            except KeyboardInterrupt:
                self._close()
                return
        if self.fs_width is None or self.fs_height is None or self.fs_framerate is None or self.fs_direction is None:
            raise ValueError("Could not initialize FlowSource metadata")
        if self.config.size is None:
            self.config.size = self.fs_width, self.fs_height
            self.logger.debug("Setting size to %dx%d", *self.config.size)

    def _setup_flow_export(self):
        if not self.export_flow:
            return
        assert self.flow_source is not None
        archive_path = self.config.get_secondary_output_path(".flow.zip")
        self.logger.debug("Setting up flow output to %s", archive_path)
        self.flow_output = NumpyOutput(archive_path, self.replace)
        self.flow_output.write_meta({
            "path": self.config.flow_path,
            "width": self.fs_width,
            "height": self.fs_height,
            "framerate": self.fs_framerate,
            "direction": self.flow_source.direction.value,
            "seek_time": self.config.seek_time,
        })

    def _setup_bitmap_source(self):
        if self.config.bitmap_path is None:
            if self.config.bitmap_alteration_path is not None:
                warnings.warn(
                    "An alteration path was passed but no bitmap was provided")
                self.logger.warning("An alteration path was passed but no bitmap was provided")
            return
        assert isinstance(self.config.size, tuple) and self.metadata_queue is not None
        self.bitmap_source = BitmapSource.from_args(
            self.config.bitmap_path,
            self.config.size,
            seek=self.ckpt_meta.get("cursor"),
            seed=self.config.seed,
            seek_time=self.config.bitmap_seek_time,
            alteration_path=self.config.bitmap_alteration_path,
            repeat=self.config.bitmap_repeat,
            flow_path=self.config.flow_path)
        self.bitmap_queue = multiprocessing.Queue(maxsize=1)
        self.bitmap_process = SourceProcess(self.bitmap_source, self.bitmap_queue, self.metadata_queue, self.log_queue, self.log_level)
        self.bitmap_process.start()
        self.logger.debug("Started bitmap process")

    def _wait_for_bitmap_source(self):
        if self.config.bitmap_path is None:
            return
        assert self.metadata_queue is not None and self.bitmap_process is not None
        while True:
            try:
                bs_width, bs_height, self.bs_framerate, self.bs_length, *_ = self.metadata_queue.get(timeout=1)
                self.logger.debug("Received metadata message from bitmap process")
                break
            except queue.Empty as exc:
                if self.bitmap_process.is_alive():
                    continue
                raise RuntimeError("Bitmap process died during initialization.") from exc
            except KeyboardInterrupt:
                self._close()
                return
        if bs_width == 0 or bs_height == 0:
            raise ValueError(
                f"Encountered an error opening bitmap '{self.config.bitmap_path}', "\
                f"shape is ({bs_height}, {bs_width})")
        if self.fs_width != bs_width or self.fs_height != bs_height:
            if bs_width % self.fs_width != 0 or bs_height % self.fs_height != 0:
                raise ValueError(
                    f"Resolutions do not match: "\
                    f"flow is {self.fs_width}x{self.fs_height} "\
                    f"while bitmap is {bs_width}x{bs_height}.")
            self.fs_width_factor = bs_width // self.fs_width
            self.fs_height_factor = bs_height // self.fs_height
            self.logger.debug("Flow and bitmap dimension do not match. Setting scaling factors to %dx%d", self.fs_width_factor, self.fs_height_factor)

    def _setup_accumulator(self):
        if self.accumulator is not None: # already loaded from checkpoint
            return
        assert self.fs_width is not None and self.fs_height is not None
        self.accumulator = Accumulator.from_args(
            int(self.fs_width * self.fs_width_factor),
            int(self.fs_height * self.fs_height_factor),
            method=self.config.acc_method,
            reset_mode=self.config.reset_mode,
            reset_alpha=self.config.reset_alpha,
            reset_mask_path=self.config.reset_mask_path,
            heatmap_mode=self.config.heatmap_mode,
            heatmap_args=self.config.heatmap_args,
            heatmap_reset_threshold=self.config.heatmap_reset_threshold,
            bg_color=self.config.accumulator_background,
            stack_composer=self.config.stack_composer,
            initial_canvas=self.config.initial_canvas,
            bitmap_mask_path=self.config.bitmap_mask_path,
            crumble=self.config.crumble,
            bitmap_introduction_flags=self.config.bitmap_introduction_flags)

    def _setup_output(self):
        if not self.has_output:
            return
        assert self.fs_width is not None and self.fs_height is not None
        vout_args = (
            int(self.fs_width * self.fs_width_factor),
            int(self.fs_height * self.fs_height_factor),
            self.bs_framerate if self.bs_framerate is not None else self.fs_framerate,
            self.config.vcodec,
            self.execute,
            self.replace,
            self.ckpt_meta.get("cursor", 0)
        )
        output_paths: list[str | None] = []
        if isinstance(self.config.output_path, list):
            output_paths += self.config.output_path
        else:
            output_paths.append(self.config.output_path)
        if self.config.output_path is not None and self.preview_output:
            output_paths.append(None)
        for i, path in enumerate(output_paths):
            output = VideoOutput.from_args(path, *vout_args)
            if output.output_path is not None:
                self.logger.debug("Output no. %d will write to %s", i, output.output_path)
            if self.export_config and output.output_path is not None:
                with pathlib.Path(output.output_path).with_suffix(".config.json").open("w") as file:
                    json.dump(self.config.todict(), file)
            oq = multiprocessing.Queue()
            oq.cancel_join_thread()
            self.output_queues.append(oq)
            op = OutputProcess(output, oq, self.log_queue, self.log_level)
            op.start()
            self.logger.debug("Started output process to %s", path)
            self.output_processes.append(op)

    def _update_flow(self) -> numpy.ndarray | None:
        assert self.flow_queue is not None
        flows = []
        for q in [self.flow_queue] + self.extra_flow_queues:
            flow = q.get(timeout=1)
            if flow is None:
                break
            flows.append(flow)
        if not flows:
            return None
        flow = self.merge_flows(flows)
        if self.fs_width_factor != 1 or self.fs_height_factor != 1:
            flow = upscale_array(flow, self.fs_width_factor, self.fs_height_factor)
        if self.flow_output is not None:
            self.flow_output.write_array(numpy.round(flow).astype(int) if self.round_flow else flow)
        return flow

    def _update_output(self, flow: numpy.ndarray) -> bool:
        assert self.accumulator is not None
        out_frame = None
        if self.config.output_intensity:
            flow_intensity = numpy.sqrt(numpy.sum(numpy.power(flow, 2), axis=2))
            out_frame = render1d(flow_intensity, self.config.render_scale, self.config.render_colors, self.config.render_binary)
        elif self.config.output_heatmap:
            out_frame = render1d(self.accumulator.get_heatmap_array(), self.config.render_scale, self.config.render_colors, self.config.render_binary)
        elif self.config.output_accumulator:
            out_frame = render2d(self.accumulator.get_accumulator_array(), self.config.render_scale, self.config.render_colors),
        elif self.bitmap_queue is not None:
            bitmap = self.bitmap_queue.get(timeout=1)
            if bitmap is None:
                return False
            out_frame = self.accumulator.apply(bitmap)
        if out_frame is not None:
            for oq in self.output_queues:
                oq.put(out_frame, timeout=1)
        return True

    def _mainloop(self):
        assert self.flow_process is not None
        assert self.accumulator is not None
        assert self.fs_direction is not None

        exception = False
        self.cursor: int = self.ckpt_meta.get("cursor", 0)
        if not isinstance(self.cursor, int):
            raise ValueError("Cursor is not an integer. Is the checkpoint valid?")
        self.logger.debug("Expected length: %s", self.expected_length)

        start_t = time.time()
        pbar = tqdm.tqdm(total=self.expected_length, unit="frame", disable=self.status_queue is not None)
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.logger.debug("Received cancel event, breaking main loop")
                break
            try:
                flow = self._update_flow()
                if flow is None:
                    break
                self.accumulator.update(flow, self.fs_direction)
                if not self._update_output(flow):
                    break
                self.cursor += 1
                if self.checkpoint_every is not None and self.cursor % self.checkpoint_every == 0:
                    self.export_checkpoint()
                pbar.update(1)
                if self.status_queue is not None:
                    self.status_queue.put(Pipeline.Status(self.cursor, self.expected_length, time.time() - start_t, None))
            except (queue.Empty, queue.Full):
                pass
            except KeyboardInterrupt:
                exception = True
                self.logger.debug("Main loop got interrupted")
                break
            except Exception as err:
                exception = True
                self.logger.error("Main loop received an exception: %s", err)
                traceback.print_exc()
                if self.status_queue is not None:
                    self.status_queue.put(Pipeline.Status(self.cursor, self.expected_length, time.time() - start_t, str(err)))
                break
            finally:
                if (not self.flow_process.is_alive())\
                    or any(not p.is_alive() for p in self.extra_flow_processes)\
                    or (self.bitmap_process is not None and not self.bitmap_process.is_alive())\
                    or any(not p.is_alive() for p in self.output_processes):
                    self.logger.debug("A child process has died, exiting main loop")
                    break
        pbar.close()
        if (exception and self.safe) or self.checkpoint_end:
            self.export_checkpoint()

    def _close(self):
        self.logger.debug("Closing pipeline")
        if self.flow_output is not None:
            self.flow_output.close()
            self.logger.debug("Closed flow output")
        if self.metadata_queue is not None:
            self.metadata_queue.close()
            self.logger.debug("Closed metadata queue")
        if self.flow_queue is not None:
            self.flow_queue.close()
            self.logger.debug("Closed flow queue")
        for i, q in enumerate(self.extra_flow_queues):
            q.close()
            self.logger.debug("Closed extra flow queue no. %d", i)
        if self.bitmap_queue is not None:
            self.bitmap_queue.close()
            self.logger.debug("Closed bitmap queue")
        for q in self.output_queues:
            q.put(None)
        if self.flow_process is not None:
            self.flow_process.kill()
            self.logger.debug("Killed flow process")
        for i, p in enumerate(self.extra_flow_processes):
            p.kill()
            self.logger.debug("Killed extra flow process no. %d", i)
        if self.bitmap_process is not None:
            self.bitmap_process.kill()
            self.logger.debug("Killed bitmap process")
        if self.flow_process is not None:
            self.flow_process.join()
            self.logger.debug("Killed flow process")
        for p in self.extra_flow_processes:
            p.join()
        if self.bitmap_process is not None:
            self.bitmap_process.join()
        for p in self.output_processes:
            p.join()
        self.logger.debug("Done closing pipeline")
        while self.log_listener is not None and self.log_listener.is_alive():
            self.log_queue.put(None)
            self.log_listener.join(timeout=.01)
        logging.shutdown()

    def run(self):
        try:
            self._setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.debug("Entering transfer function")
            self._setup_checkpoint()
            if not (self.has_output or self.export_flow or self.checkpoint_end):
                warnings.warn("No output or exportation selected")
            self.metadata_queue = multiprocessing.Queue()
            self._setup_flow_sources()
            self._wait_for_flow_sources()
            self._setup_flow_export()
            self._setup_bitmap_source()
            self._wait_for_bitmap_source()
            self.metadata_queue.close()
            self.logger.debug("Closed metadata queue")
            self._setup_accumulator()
            self._setup_output()
            if self.safe:
                with open("last-config.json", "w") as file:
                    json.dump(self.config.todict(), file)
            self._mainloop()
        except Exception as err:
            self.logger.error("Pipeline encountered an error: %s", err)
            traceback.print_exc()
        finally:
            self._close()
