import multiprocessing
import os
import queue
import random
import re
import time
import traceback
import warnings

import numpy
import tqdm

from .flow import FlowSource, FlowDirection
from .bitmap import BitmapSource
from .accumulator import Accumulator
from .output import VideoOutput, ZipOutput, NumpyOutput, render1d, render2d


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
                    self.source.length if is_fs else None,
                    self.source.seek if is_fs else None,
                    self.source.duration if is_fs else None
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


def transfer(
        flow_path: str,
        bitmap_path: str | None,
        output_path: str | None,
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
        stack_background: str = "ffffff",
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
        safe: bool = True,
        seed: int | None = None,
        seek_time: float | None = None,
        bitmap_seek_time: float | None = None,
        duration_time: float | None = None):

    if safe:
        append_history()

    shape_queue = flow_queue = bitmap_queue = flow_process = bitmap_process\
        = output_queue = output_process = flow_output = bs_framerate\
        = accumulator = None

    def close():
        if export_flow:
            flow_output.close()
        if shape_queue is not None:
            shape_queue.close()
        if flow_queue is not None:
            flow_queue.close()
        if bitmap_queue is not None:
            bitmap_queue.close()
        if output_queue is not None:
            output_queue.put(None)
        if flow_process is not None:
            flow_process.kill()
        if bitmap_process is not None:
            bitmap_process.kill()
        if flow_process is not None:
            flow_process.join()
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

        if not (has_output or export_flow):
            warnings.warn("No output or exportation selected")

        if size is not None:
            size = tuple(map(int, re.split(r"[^\d]", size)))

        flow_source = FlowSource.from_args(
            flow_path, use_mvs=use_mvs, mask_path=mask_path,
            kernel_path=kernel_path, cv_config=cv_config, flow_gain=flow_gain,
            size=size, direction=direction, seek=ckpt_meta.get("cursor"),
            seek_time=seek_time, duration_time=duration_time)

        shape_queue = multiprocessing.Queue()

        flow_queue = multiprocessing.Queue(maxsize=1)
        flow_process = SourceProcess(flow_source, flow_queue, shape_queue)
        flow_process.start()
        while True:
            try:
                (fs_width, fs_height, fs_framerate, fs_direction, fs_length,
                 fs_seek, fs_duration) = shape_queue.get(timeout=1)
                break
            except queue.Empty as exc:
                if flow_process.is_alive():
                    continue
                raise RuntimeError("Flow process died during initialization.") from exc

        if export_flow:
            archive_path = get_secondary_output_path(flow_path, output_path, ".flow.zip")
            flow_output = NumpyOutput(archive_path, replace)
            flow_output.write_meta({
                "path": flow_path,
                "width": fs_width,
                "height": fs_height,
                "framerate": fs_framerate,
                "direction": flow_source.direction.value,
                "length": fs_length,
                "seek": fs_seek,
                "duration": fs_duration
            })

        if size is None:
            size = fs_width, fs_height

        if output_bitmap:
            bitmap_source = BitmapSource.from_args(
                bitmap_path, size, seek=ckpt_meta.get("cursor"), seed=seed,
                seek_time=bitmap_seek_time)
            bitmap_queue = multiprocessing.Queue(maxsize=1)
            bitmap_process = SourceProcess(bitmap_source, bitmap_queue, shape_queue)
            bitmap_process.start()
            bs_width, bs_height, bs_framerate, *_ = shape_queue.get()

            if fs_width != bs_width or fs_height != bs_height:
                raise ValueError(f"Resolutions do not match: flow is {fs_width}x{fs_height} "\
                                 f"while bitmap is {bs_width}x{bs_height}.")

        shape_queue.close()

        if accumulator is None:
            accumulator = Accumulator.from_args(fs_width, fs_height, acc_method,
                reset_mode, reset_alpha, reset_mask_path, heatmap_mode,
                heatmap_args, stack_background, stack_composer)

        if has_output:
            output = VideoOutput.from_args(
                output_path, fs_width, fs_height,
                bs_framerate if bs_framerate is not None else fs_framerate,
                vcodec, execute, replace, safe)

            output_queue = multiprocessing.Queue()
            output_process = OutputProcess(output, output_queue)
            output_process.start()

        exception = False
        cursor = ckpt_meta.get("cursor", 0)
        pbar = tqdm.tqdm(total=fs_duration, unit="frame")
        while True:
            try:
                flow = flow_queue.get(timeout=1)
                if flow is None:
                    break
                if export_flow:
                    flow_output.write_array(numpy.round(flow).astype(int) if round_flow else flow)
                accumulator.update(flow, fs_direction)
                if output_bitmap:
                    bitmap = bitmap_queue.get(timeout=1)
                    if bitmap is None:
                        break
                    output_queue.put(accumulator.apply(bitmap), timeout=1)
                elif output_intensity:
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
                cursor += 1
                if checkpoint_every is not None and cursor % checkpoint_every == 0:
                    export_checkpoint(flow_path, bitmap_path, output_path,
                                      replace, cursor, accumulator, seed)
                pbar.update(1)
            except (queue.Empty, queue.Full):
                if (not flow_process.is_alive())\
                    or (bitmap_process is not None and not bitmap_process.is_alive())\
                    or (output_process is not None and not output_process.is_alive()):
                    exception = True
                    break
            except KeyboardInterrupt:
                exception = True
                break
            except Exception:
                exception = True
                traceback.print_exc()
                break
        pbar.close()
        if exception and safe:
            export_checkpoint(flow_path, bitmap_path, output_path, replace,
                              cursor, accumulator, seed)
        close()

    except Exception as err:
        close()
        raise err
