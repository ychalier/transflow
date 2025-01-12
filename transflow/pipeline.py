import multiprocessing
import os
import queue
import random
import re
import time
import traceback
import typing
import warnings

import numpy
import tqdm

from .flow import FlowSource, FlowDirection
from .bitmap import BitmapSource
from .accumulator import Accumulator
from .output import VideoOutput, ZipOutput, NumpyOutput, render1d, render2d
from .utils import multiply_arrays, binarize_arrays, absmax


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
                is_fs = isinstance(self.source, FlowSource)
                self.shape_queue.put((
                    self.source.width,
                    self.source.height,
                    self.source.framerate,
                    self.source.direction if is_fs else None,
                    self.source.length if is_fs else None
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
                    continue
                except Exception:
                    traceback.print_exc()
                    break


def get_secondary_output_path(
        flow_path: str,
        output_path: str | None,
        suffix: str) -> str:
    path = os.path.splitext(flow_path if output_path is None else output_path)[0]
    if path.endswith(".flow") or path.endswith(".ckpt"):
        path = path[:-5]
    if re.match(r".*\.(\d{3})$", path):
        path = path[:-4]
    return path + suffix


def export_checkpoint(
        flow_path: str,
        bitmap_path: str | None,
        output_path: str | None,
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
    "sum": sum,
    "average": lambda flows: sum(flows) / len(flows),
    "difference": lambda flows: flows[0] - sum(flows[1:]),
    "product": multiply_arrays,
    "maskbin": lambda flows: multiply_arrays([flows[0]] + binarize_arrays(flows[1:])),
    "masklin": lambda flows: multiply_arrays([flows[0]] + [numpy.abs(f) for f in flows[1:]]),
    "absmax": absmax,
}


def upscale_flow(flow: numpy.ndarray, wf: int, hf: int) -> numpy.ndarray:
    return numpy.kron(flow * (wf, hf), numpy.ones((hf, wf, 1))).astype(flow.dtype)


def transfer(
        flow_path: str,
        bitmap_path: str | None,
        output_path: str | None,
        extra_flow_paths: list[str] | None,
        flows_merging_function: str = "first",
        use_mvs: bool = False,
        vcodec: str = "h264",
        reset_mode: str = "off",
        reset_alpha: float = .9,
        heatmap_mode: str = "discrete",
        heatmap_args: str = "0:4:2:1",
        execute: bool = False,
        replace: bool = False,
        mask_path: str | None = None,
        kernel_path: str | None = None,
        reset_mask_path: str | None = None,
        cv_config: str | None = None,
        flow_gain: str | None = None,
        size: str | None = None,
        acc_method: str = "map",
        accumulator_background: str = "ffffff",
        stack_composer: str = "top",
        direction: str = "forward",
        round_flow: bool = False,
        export_flow: bool = False,
        output_intensity: bool = False,
        output_heatmap: bool = False,
        output_accumulator: bool = False,
        render_scale: float = 1,
        render_colors: str | None = None,
        render_binary: bool = False,
        checkpoint_every: int | None = None,
        checkpoint_end: bool = False,
        safe: bool = True,
        seed: int | None = None,
        seek_time: float | None = None,
        bitmap_seek_time: float | None = None,
        duration_time: float | None = None,
        bitmap_alteration_path: str | None = None,
        repeat: int = 1,
        initial_canvas: str | None = None,
        bitmap_mask_path: str | None = None,
        crumble: bool = False,
        bitmap_introduction_flags: int = 1,
        initially_crumbled: bool = False):

    if safe:
        append_history()

    shape_queue = flow_queue = bitmap_queue = flow_process = bitmap_process\
        = output_queue = output_process = flow_output = bs_framerate\
        = accumulator = None

    if extra_flow_paths is None:
        extra_flow_paths: list[str] = []
        flows_merging_function = "first"
    extra_flow_sources: list[FlowSource] = []
    extra_flow_queues: list[multiprocessing.Queue] = []
    extra_flow_processes: list[SourceProcess] = []
    merge_flows = flows_merging_functions[flows_merging_function]

    def close():
        if export_flow:
            flow_output.close()
        if shape_queue is not None:
            shape_queue.close()
        if flow_queue is not None:
            flow_queue.close()
        for q in extra_flow_queues:
            q.close()
        if bitmap_queue is not None:
            bitmap_queue.close()
        if output_queue is not None:
            output_queue.put(None)
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
        if output_process is not None:
            output_process.join()

    try:

        ckpt_meta = {}
        if flow_path.endswith(".ckpt.zip"):
            import json, pickle, zipfile
            with zipfile.ZipFile(flow_path) as archive:
                with archive.open("meta.json") as file:
                    ckpt_meta = json.loads(file.read().decode())
                with archive.open("accumulator.bin") as file:
                    accumulator = pickle.load(file)
            flow_path = ckpt_meta["flow_path"]
            seed = ckpt_meta["seed"]

        if seed is None:
            seed = random.randint(0, 2**32-1)

        if direction == "forward":
            direction = FlowDirection.FORWARD
        elif direction == "backward":
            direction = FlowDirection.BACKWARD
        else:
            raise ValueError(f"Invalid flow direction '{direction}'")

        if render_colors is not None:
            render_colors = render_colors.split(",")

        output_bitmap = bitmap_path is not None
        has_output = output_bitmap or output_intensity or output_heatmap\
            or output_accumulator

        if not (has_output or export_flow or checkpoint_end):
            warnings.warn("No output or exportation selected")

        if size is not None:
            size = tuple(map(int, re.split(r"[^\d]", size)))

        flow_source = FlowSource.from_args(
            flow_path, use_mvs=use_mvs, mask_path=mask_path,
            kernel_path=kernel_path, cv_config=cv_config, flow_gain=flow_gain,
            size=size, direction=direction, seek_ckpt=ckpt_meta.get("cursor"),
            seek_time=seek_time, duration_time=duration_time, repeat=repeat)

        shape_queue = multiprocessing.Queue()

        flow_queue = multiprocessing.Queue(maxsize=1)
        flow_process = SourceProcess(flow_source, flow_queue, shape_queue)
        flow_process.start()

        for extra_flow_path in extra_flow_paths:
            extra_flow_sources.append(FlowSource.from_args(
                extra_flow_path, use_mvs=use_mvs, mask_path=mask_path,
                kernel_path=kernel_path, cv_config=cv_config,
                flow_gain=flow_gain, size=size, direction=direction,
                seek_ckpt=ckpt_meta.get("cursor"), seek_time=seek_time,
                duration_time=duration_time))
            extra_flow_queues.append(multiprocessing.Queue(maxsize=1))
            extra_flow_processes.append(SourceProcess(
                extra_flow_sources[-1], extra_flow_queues[-1], shape_queue))
            extra_flow_processes[-1].start()

        flow_sources_to_load = 1 + len(extra_flow_processes)
        flow_sources_loaded = 0

        while True:
            try:
                shape_info = shape_queue.get(timeout=1)
                if flow_sources_loaded == 0:
                    (fs_width, fs_height, fs_framerate, fs_direction, fs_length) = shape_info
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

        if export_flow:
            archive_path = get_secondary_output_path(flow_path, output_path, ".flow.zip")
            flow_output = NumpyOutput(archive_path, replace)
            flow_output.write_meta({
                "path": flow_path,
                "width": fs_width,
                "height": fs_height,
                "framerate": fs_framerate,
                "direction": flow_source.direction.value,
                "seek_time": seek_time,
            })

        if size is None:
            size = fs_width, fs_height
        
        fs_width_factor = fs_height_factor = 1

        if output_bitmap:
            bitmap_source = BitmapSource.from_args(
                bitmap_path,
                size,
                seek=ckpt_meta.get("cursor"),
                seed=seed,
                seek_time=bitmap_seek_time,
                alteration_path=bitmap_alteration_path)
            bitmap_queue = multiprocessing.Queue(maxsize=1)
            bitmap_process = SourceProcess(bitmap_source, bitmap_queue, shape_queue)
            bitmap_process.start()

            while True:
                try:
                    bs_width, bs_height, bs_framerate, *_ = shape_queue.get(timeout=1)
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
                    f"Encountered an error opening bitmap '{bitmap_path}', "\
                    f"shape is ({bs_height}, {bs_width})")

            if fs_width != bs_width or fs_height != bs_height:
                if bs_width % fs_width != 0 or bs_height % fs_height != 0:
                    raise ValueError(
                        f"Resolutions do not match: "\
                        f"flow is {fs_width}x{fs_height} "\
                        f"while bitmap is {bs_width}x{bs_height}.")
                fs_width_factor = bs_width // fs_width
                fs_height_factor = bs_height // fs_height
        
        elif bitmap_alteration_path is not None:
            warnings.warn(
                "An alteration path was passed but no bitmap was provided")

        shape_queue.close()

        if accumulator is None:
            accumulator = Accumulator.from_args(
                fs_width * fs_width_factor, 
                fs_height * fs_height_factor,
                acc_method,
                reset_mode,
                reset_alpha,
                reset_mask_path,
                heatmap_mode,
                heatmap_args,
                accumulator_background,
                stack_composer,
                initial_canvas,
                bitmap_mask_path,
                crumble,
                bitmap_introduction_flags,
                initially_crumbled)

        if has_output:
            output = VideoOutput.from_args(
                output_path,
                fs_width * fs_width_factor,
                fs_height * fs_height_factor,
                bs_framerate if bs_framerate is not None else fs_framerate,
                vcodec,
                execute,
                replace,
                safe)

            output_queue = multiprocessing.Queue()
            output_queue.cancel_join_thread()
            output_process = OutputProcess(output, output_queue)
            output_process.start()

        exception = False
        cursor = ckpt_meta.get("cursor", 0)
        pbar = tqdm.tqdm(
            total=None if fs_length is None else fs_length - cursor,
            unit="frame")
        while True:
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
                if export_flow:
                    flow_output.write_array(numpy.round(flow).astype(int) if round_flow else flow)
                accumulator.update(flow, fs_direction)
                if output_intensity:
                    flow_intensity = numpy.sqrt(numpy.sum(numpy.power(flow, 2), axis=2))
                    output_queue.put(
                        render1d(flow_intensity, render_scale, render_colors,
                                 render_binary),
                        timeout=1)
                elif output_heatmap:
                    output_queue.put(
                        render1d(accumulator.get_heatmap_array(), render_scale,
                                 render_colors, render_binary),
                        timeout=1)
                elif output_accumulator:
                    output_queue.put(
                        render2d(accumulator.get_accumulator_array(),
                                 render_scale, render_colors),
                        timeout=1)
                elif output_bitmap:
                    bitmap = bitmap_queue.get(timeout=1)
                    if bitmap is None:
                        break
                    output_queue.put(accumulator.apply(bitmap), timeout=1)
                cursor += 1
                if checkpoint_every is not None and cursor % checkpoint_every == 0:
                    export_checkpoint(flow_path, bitmap_path, output_path,
                                      replace, cursor, accumulator, seed)
                pbar.update(1)
            except (queue.Empty, queue.Full):
                pass
            except KeyboardInterrupt:
                exception = True
                break
            except Exception:
                exception = True
                traceback.print_exc()
                break
            finally:
                if (not flow_process.is_alive())\
                    or (bitmap_process is not None and not bitmap_process.is_alive())\
                    or (output_process is not None and not output_process.is_alive()):
                    exception = True
                    break
        pbar.close()
        if (exception and safe) or checkpoint_end:
            export_checkpoint(flow_path, bitmap_path, output_path, replace,
                              cursor, accumulator, seed)
        close()

    except Exception as err:
        close()
        raise err
