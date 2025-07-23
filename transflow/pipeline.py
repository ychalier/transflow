import dataclasses
import multiprocessing
import multiprocessing.queues
import os
import queue
import random
import re
import threading
import time
import traceback
import typing
import warnings

import numpy
import tqdm

from .flow import FlowSource, FlowDirection, LockMode
from .bitmap import BitmapSource
from .accumulator import Accumulator
from .output import VideoOutput, ZipOutput, NumpyOutput
from .utils import multiply_arrays, binarize_arrays, absmax, parse_hex_color


def render1d(arr: numpy.ndarray, scale: float = 1,
             colors: tuple[str, ...] | None = None, binary: bool = False
             ) -> numpy.ndarray:
    if colors is None:
        colors = ("#000000", "#ffffff")
    color_arrs = [numpy.array(parse_hex_color(c), dtype=numpy.float32) for c in colors]
    out_shape = (*arr.shape[:2], 1)
    if binary:
        coeff = numpy.clip(numpy.round(scale * arr), 0, 1).reshape(out_shape)
        coeff_a = 1 - coeff
        coeff_b = coeff
    else:
        coeff_a = numpy.clip(1 - scale * arr, 0, 1).reshape(out_shape)
        coeff_b = numpy.clip(scale * arr, 0, 1).reshape(out_shape)
    frame = numpy.multiply(coeff_a, color_arrs[0]) + numpy.multiply(coeff_b, color_arrs[1])
    return numpy.clip(frame, 0, 255).astype(numpy.uint8)


def render2d(arr: numpy.ndarray, scale: float = 1,
              colors: tuple[str, ...] | None = None)-> numpy.ndarray:
    if colors is None:
        colors = ("#ffff00", "#0000ff", "#ff00ff", "#00ff00")
    color_arrs = [numpy.array(parse_hex_color(c), dtype=numpy.float32) for c in colors]
    out_shape = (*arr.shape[:2], 1)
    coeff_y = numpy.clip(1 + scale * arr[:,:,0], 0, 1).reshape(out_shape)
    coeff_b = numpy.clip(1 - scale * arr[:,:,0], 0, 1).reshape(out_shape)
    coeff_m = numpy.clip(1 + scale * arr[:,:,1], 0, 1).reshape(out_shape)
    coeff_g = numpy.clip(1 - scale * arr[:,:,1], 0, 1).reshape(out_shape)
    frame = .5 * (
        numpy.multiply(coeff_y, color_arrs[0])
        + numpy.multiply(coeff_b, color_arrs[1])
        + numpy.multiply(coeff_m, color_arrs[2])
        + numpy.multiply(coeff_g, color_arrs[3]))
    return numpy.clip(frame, 0, 255).astype(numpy.uint8)



def append_history():
    import datetime, sys
    f = lambda s: os.path.realpath(s) if os.path.isfile(s) else s
    line = " ".join([
        datetime.datetime.now().isoformat(),
        *list(map(f, sys.argv))])
    with open("history.log", "a", encoding="utf8") as file:
        file.write("\n" + line + "\n")


class SourceProcess(multiprocessing.Process):

    def __init__(self, source: FlowSource | BitmapSource,
                 q: multiprocessing.Queue, sq: multiprocessing.Queue):
        multiprocessing.Process.__init__(self)
        self.source = source
        self.queue = q
        self.shape_queue = sq

    def run(self):
        put_none = True
        try:
            with self.source:
                self.shape_queue.put((
                    self.source.width,
                    self.source.height,
                    self.source.framerate,
                    self.source.length,
                    self.source.direction if isinstance(self.source, FlowSource) else None
                ))
                try:
                    for item in self.source:
                        self.queue.put(item)
                except KeyboardInterrupt:
                    put_none = False
                except Exception:
                    put_none = False
                    traceback.print_exc()
        except Exception:
            put_none = False
            traceback.print_exc()
        if put_none:
            self.queue.put(None)


class OutputProcess(multiprocessing.Process):

    def __init__(self, output: VideoOutput, q: multiprocessing.Queue):
        multiprocessing.Process.__init__(self)
        self.output = output
        self.queue = q

    def run(self):
        with self.output:
            while True:
                try:
                    frame = self.queue.get()
                    if frame is None:
                        break
                    self.output.feed(frame)
                except KeyboardInterrupt:
                    break # was 'continue' before, but why?
                except Exception:
                    traceback.print_exc()
                    break


def get_secondary_output_path(
        flow_path: str,
        output_path: str | list[str] | None,
        suffix: str) -> str:
    base_output_path = None
    if isinstance(output_path, list):
        mjpeg_pattern = re.compile(r"^mjpeg(:[:a-z0-9A-Z\-]+)?$", re.IGNORECASE)
        for path in output_path:
            if mjpeg_pattern.match(path):
                continue
            base_output_path = path
            break
    else:
        base_output_path = output_path
    path = os.path.splitext(flow_path if base_output_path is None else base_output_path)[0]
    if path.endswith(".flow") or path.endswith(".ckpt"):
        path = path[:-5]
    if re.match(r".*\.(\d{3})$", path):
        path = path[:-4]
    return path + suffix


def export_checkpoint(
        flow_path: str,
        bitmap_path: str | None,
        output_path: str | list[str] | None,
        replace: bool,
        cursor: int,
        accumulator: Accumulator,
        seed: int):
    output = ZipOutput(
        get_secondary_output_path(flow_path, output_path, f"_{cursor:05d}.ckpt.zip"),
        replace)
    output.write_meta({
        "flow_path": flow_path,
        "bitmap_path": bitmap_path,
        "output_path": output_path,
        "cursor": cursor,
        "timestamp": time.time(),
        "seed": seed
    })
    output.write_object("accumulator.bin", accumulator)
    output.close()


flows_merging_functions: dict[str, typing.Callable[[list[numpy.ndarray]], numpy.ndarray]] = {
    "first": lambda flows: flows[0],
    "sum": lambda flows: numpy.sum(flows, axis=0),
    "average": lambda flows: numpy.sum(flows, axis=0) / len(flows),
    "difference": lambda flows: flows[0] - sum(flows[1:]),
    "product": multiply_arrays,
    "maskbin": lambda flows: multiply_arrays([flows[0]] + binarize_arrays(flows[1:])),
    "masklin": lambda flows: multiply_arrays([flows[0]] + [numpy.abs(f) for f in flows[1:]]),
    "absmax": absmax,
}


def upscale_flow(flow: numpy.ndarray, wf: int, hf: int) -> numpy.ndarray:
    return numpy.kron(flow * (wf, hf), numpy.ones((hf, wf, 1))).astype(flow.dtype)


def get_expected_length(fs_length: int | None, bs_length: int | None, cursor: int) -> int | None:
    expected_length = None
    if fs_length is not None and bs_length is not None:
        expected_length = min(fs_length, bs_length)
    elif fs_length is not None:
        expected_length = fs_length
    elif bs_length is not None:
        expected_length = bs_length
    if expected_length is not None:
        expected_length -= cursor
    return expected_length


@dataclasses.dataclass
class Config:

    # Positional Args
    flow_path: str
    bitmap_path: str | None
    output_path: str | list[str] | None
    extra_flow_paths: list[str] | None

    # Flow Args
    flows_merging_function: str = "first"
    use_mvs: bool = False
    mask_path: str | None = None
    kernel_path: str | None = None
    cv_config: str | None = None
    flow_filters: str | None = None
    direction: str | FlowDirection = "forward"
    round_flow: bool = False
    export_flow: bool = False
    seek_time: float | None = None
    duration_time: float | None = None
    repeat: int = 1
    lock_expr: str | None = None
    lock_mode: str | LockMode = LockMode.STAY

    # Bitmap Args
    bitmap_seek_time: float | None = None
    bitmap_alteration_path: str | None = None
    bitmap_repeat: int = 1

    # Accumulator Args
    reset_mode: str = "off"
    reset_alpha: float = .9
    reset_mask_path: str | None = None
    heatmap_mode: str = "discrete"
    heatmap_args: str = "0:4:2:1"
    heatmap_reset_threshold: float | None = None
    acc_method: str = "map"
    accumulator_background: str = "ffffff"
    stack_composer: str = "top"
    initial_canvas: str | None = None
    bitmap_mask_path: str | None = None
    crumble: bool = False
    bitmap_introduction_flags: int = 1

    # Output Args
    vcodec: str = "h264"
    execute: bool = False
    replace: bool = False
    size: str | tuple[int, int] | None = None
    output_intensity: bool = False
    output_heatmap: bool = False
    output_accumulator: bool = False
    render_scale: float = 1
    render_colors: str | tuple[str, ...] | None = None
    render_binary: bool = False
    checkpoint_every: int | None = None
    checkpoint_end: bool = False
    safe: bool = True
    preview_output: bool = False

    # General Args
    seed: int | None = None


@dataclasses.dataclass
class Status:

    cursor: int
    total: int | None
    elapsed: float
    error: str | None


def transfer(
        config: Config,
        cancel_event: threading.Event | None = None,
        status_queue: multiprocessing.queues.Queue | None = None):

    if config.safe:
        append_history()

    shape_queue = flow_queue = bitmap_queue = flow_process = bitmap_process\
        = flow_output = bs_framerate = bs_length = accumulator = None
    output_queues: list[multiprocessing.Queue] = []
    output_processes: list[OutputProcess] = []

    if config.extra_flow_paths is None:
        config.extra_flow_paths = []
        config.flows_merging_function = "first"
    extra_flow_sources: list[FlowSource] = []
    extra_flow_queues: list[multiprocessing.Queue] = []
    extra_flow_processes: list[SourceProcess] = []
    merge_flows = flows_merging_functions[config.flows_merging_function]

    def close():
        if flow_output is not None:
            flow_output.close()
        if shape_queue is not None:
            shape_queue.close()
        if flow_queue is not None:
            flow_queue.close()
        for q in extra_flow_queues:
            q.close()
        if bitmap_queue is not None:
            bitmap_queue.close()
        for q in output_queues:
            q.put(None)
        if flow_process is not None:
            flow_process.kill()
        for p in extra_flow_processes:
            p.kill()
        if bitmap_process is not None:
            bitmap_process.kill()
        if flow_process is not None:
            flow_process.join()
        for p in extra_flow_processes:
            p.join()
        if bitmap_process is not None:
            bitmap_process.join()
        for p in output_processes:
            p.join()

    try:

        ckpt_meta = {}
        if config.flow_path.endswith(".ckpt.zip"):
            import json, pickle, zipfile
            with zipfile.ZipFile(config.flow_path) as archive:
                with archive.open("meta.json") as file:
                    ckpt_meta = json.loads(file.read().decode())
                with archive.open("accumulator.bin") as file:
                    accumulator = pickle.load(file)
            config.flow_path = ckpt_meta["flow_path"]
            config.seed = ckpt_meta["seed"]

        if config.seed is None:
            config.seed = random.randint(0, 2**32-1)

        if config.direction == "forward":
            config.direction = FlowDirection.FORWARD
        elif config.direction == "backward":
            config.direction = FlowDirection.BACKWARD
        else:
            raise ValueError(f"Invalid flow direction '{config.direction}'")

        if isinstance(config.render_colors, str):
            config.render_colors = tuple(config.render_colors.split(","))

        has_output = config.bitmap_path is not None or config.output_intensity or config.output_heatmap\
            or config.output_accumulator

        if not (has_output or config.export_flow or config.checkpoint_end):
            warnings.warn("No output or exportation selected")

        if isinstance(config.size, str):
            width, height = tuple(map(int, re.split(r"[^\d]", config.size)))
            config.size = width, height

        fs_args = {
            "use_mvs": config.use_mvs,
            "mask_path": config.mask_path,
            "kernel_path": config.kernel_path,
            "cv_config": config.cv_config,
            "flow_filters": config.flow_filters,
            "size": config.size,
            "direction": config.direction,
            "seek_ckpt": ckpt_meta.get("cursor"),
            "seek_time": config.seek_time,
            "duration_time": config.duration_time,
            "repeat": config.repeat,
            "lock_expr": config.lock_expr,
            "lock_mode": config.lock_mode
        }

        flow_source = FlowSource.from_args(config.flow_path, **fs_args)

        shape_queue = multiprocessing.Queue()

        flow_queue = multiprocessing.Queue(maxsize=1)
        flow_process = SourceProcess(flow_source, flow_queue, shape_queue)
        flow_process.start()

        for extra_flow_path in config.extra_flow_paths:
            extra_flow_sources.append(FlowSource.from_args(extra_flow_path, **fs_args))
            extra_flow_queues.append(multiprocessing.Queue(maxsize=1))
            extra_flow_processes.append(SourceProcess(
                extra_flow_sources[-1], extra_flow_queues[-1], shape_queue))
            extra_flow_processes[-1].start()

        flow_sources_to_load = 1 + len(extra_flow_processes)
        flow_sources_loaded = 0

        fs_width = fs_height = fs_framerate = fs_length = fs_direction = None

        while True:
            try:
                shape_info = shape_queue.get(timeout=1)
                if flow_sources_loaded == 0:
                    (fs_width, fs_height, fs_framerate, fs_length, fs_direction) = shape_info
                flow_sources_loaded += 1
                if flow_sources_loaded >= flow_sources_to_load:
                    break
            except queue.Empty as exc:
                if flow_process.is_alive() and all(p.is_alive() for p in extra_flow_processes):
                    continue
                raise RuntimeError("Flow process died during initialization.") from exc
            except KeyboardInterrupt:
                close()
                return

        if fs_width is None or fs_height is None or fs_framerate is None or fs_direction is None:
            raise ValueError("Could not initialize FlowSource metadata")

        if config.export_flow:
            archive_path = get_secondary_output_path(config.flow_path, config.output_path, ".flow.zip")
            flow_output = NumpyOutput(archive_path, config.replace)
            flow_output.write_meta({
                "path": config.flow_path,
                "width": fs_width,
                "height": fs_height,
                "framerate": fs_framerate,
                "direction": flow_source.direction.value,
                "seek_time": config.seek_time,
            })

        if config.size is None:
            config.size = fs_width, fs_height

        fs_width_factor = fs_height_factor = 1

        if config.bitmap_path is not None:
            bitmap_source = BitmapSource.from_args(
                config.bitmap_path,
                config.size,
                seek=ckpt_meta.get("cursor"),
                seed=config.seed,
                seek_time=config.bitmap_seek_time,
                alteration_path=config.bitmap_alteration_path,
                repeat=config.bitmap_repeat,
                flow_path=config.flow_path)
            bitmap_queue = multiprocessing.Queue(maxsize=1)
            bitmap_process = SourceProcess(bitmap_source, bitmap_queue, shape_queue)
            bitmap_process.start()

            while True:
                try:
                    bs_width, bs_height, bs_framerate, bs_length, *_ = shape_queue.get(timeout=1)
                    break
                except queue.Empty as exc:
                    if bitmap_process.is_alive():
                        continue
                    raise RuntimeError("Bitmap process died during initialization.") from exc
                except KeyboardInterrupt:
                    close()
                    return

            if bs_width == 0 or bs_height == 0:
                raise ValueError(
                    f"Encountered an error opening bitmap '{config.bitmap_path}', "\
                    f"shape is ({bs_height}, {bs_width})")

            if fs_width != bs_width or fs_height != bs_height:
                if bs_width % fs_width != 0 or bs_height % fs_height != 0:
                    raise ValueError(
                        f"Resolutions do not match: "\
                        f"flow is {fs_width}x{fs_height} "\
                        f"while bitmap is {bs_width}x{bs_height}.")
                fs_width_factor = bs_width // fs_width
                fs_height_factor = bs_height // fs_height

        elif config.bitmap_alteration_path is not None:
            warnings.warn(
                "An alteration path was passed but no bitmap was provided")

        shape_queue.close()

        if accumulator is None:
            accumulator = Accumulator.from_args(
                fs_width * fs_width_factor,
                fs_height * fs_height_factor,
                method=config.acc_method,
                reset_mode=config.reset_mode,
                reset_alpha=config.reset_alpha,
                reset_mask_path=config.reset_mask_path,
                heatmap_mode=config.heatmap_mode,
                heatmap_args=config.heatmap_args,
                heatmap_reset_threshold=config.heatmap_reset_threshold,
                bg_color=config.accumulator_background,
                stack_composer=config.stack_composer,
                initial_canvas=config.initial_canvas,
                bitmap_mask_path=config.bitmap_mask_path,
                crumble=config.crumble,
                bitmap_introduction_flags=config.bitmap_introduction_flags)

        if has_output:
            vout_args = (
                fs_width * fs_width_factor,
                fs_height * fs_height_factor,
                bs_framerate if bs_framerate is not None else fs_framerate,
                config.vcodec,
                config.execute,
                config.replace,
                config.safe
            )
            if isinstance(config.output_path, list) and not config.output_path:
                config.output_path = None
            output_paths: list[str | None] = []
            if isinstance(config.output_path, list):
                output_paths += config.output_path
            else:
                output_paths.append(config.output_path)
            if config.output_path is not None and config.preview_output:
                output_paths.append(None)
            for path in output_paths:
                output = VideoOutput.from_args(path, *vout_args)
                oq = multiprocessing.Queue()
                oq.cancel_join_thread()
                output_queues.append(oq)
                op = OutputProcess(output, oq)
                op.start()
                output_processes.append(op)

        exception = False
        cursor = ckpt_meta.get("cursor", 0)
        expected_length = get_expected_length(fs_length, bs_length, cursor)

        start_t = time.time()
        pbar = tqdm.tqdm(total=expected_length, unit="frame", disable=status_queue is not None)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                flows = []
                break_now = False
                for q in [flow_queue] + extra_flow_queues:
                    flow = q.get(timeout=1)
                    if flow is None:
                        break_now = True
                        break
                    flows.append(flow)
                if break_now:
                    break
                flow = merge_flows(flows)
                if fs_width_factor != 1 or fs_height_factor != 1:
                    flow = upscale_flow(flow, fs_width_factor, fs_height_factor)
                if flow_output is not None:
                    flow_output.write_array(numpy.round(flow).astype(int) if config.round_flow else flow)
                accumulator.update(flow, fs_direction)
                out_frame = None
                if config.output_intensity:
                    flow_intensity = numpy.sqrt(numpy.sum(numpy.power(flow, 2), axis=2))
                    out_frame = render1d(flow_intensity, config.render_scale, config.render_colors, config.render_binary)
                elif config.output_heatmap:
                    out_frame = render1d(accumulator.get_heatmap_array(), config.render_scale, config.render_colors, config.render_binary)
                elif config.output_accumulator:
                    out_frame = render2d(accumulator.get_accumulator_array(), config.render_scale, config.render_colors),
                elif bitmap_queue is not None:
                    bitmap = bitmap_queue.get(timeout=1)
                    if bitmap is None:
                        break
                    out_frame = accumulator.apply(bitmap)
                if out_frame is not None:
                    for oq in output_queues:
                        oq.put(out_frame, timeout=1)
                cursor += 1
                if config.checkpoint_every is not None and cursor % config.checkpoint_every == 0:
                    export_checkpoint(config.flow_path, config.bitmap_path, config.output_path,
                                      config.replace, cursor, accumulator, config.seed)
                pbar.update(1)
                if status_queue is not None:
                    status_queue.put(Status(cursor, expected_length, time.time() - start_t, None))
            except (queue.Empty, queue.Full):
                pass
            except KeyboardInterrupt:
                exception = True
                break
            except Exception as err:
                exception = True
                traceback.print_exc()
                if status_queue is not None:
                    status_queue.put(Status(cursor, expected_length, time.time() - start_t, str(err)))
                break
            finally:
                if (not flow_process.is_alive())\
                    or any(not p.is_alive() for p in extra_flow_processes)\
                    or (bitmap_process is not None and not bitmap_process.is_alive())\
                    or any(not p.is_alive() for p in output_processes):
                    exception = True
                    break
        pbar.close()
        if (exception and config.safe) or config.checkpoint_end:
            export_checkpoint(config.flow_path, config.bitmap_path, config.output_path, config.replace,
                              cursor, accumulator, config.seed)
        close()

    except Exception as err:
        close()
        raise err
