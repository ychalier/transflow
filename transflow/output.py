import os
import re
import subprocess

import cv2
import numpy

from .mjpeg import MjpegStream, MjpegServer
from .utils import parse_hex_color, find_unique_path, startfile


def append_history(output_path):
    with open("history.log", "a", encoding="utf8") as file:
        file.write(f"└► {os.path.realpath(output_path)}\n")


class VideoOutput:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __enter__(self):
        return self

    def feed(self, frame: numpy.ndarray):
        raise NotImplementedError()

    def __close__(self):
        pass

    @classmethod
    def from_args(cls, path: str | None, width: int, height: int,
                  framerate: int | None = None, vcodec: str = "h264",
                  execute: bool = False, replace: bool = False,
                  safe: bool = False):
        if path is None:
            return CvVideoOutput(width, height)
        mjpeg_pattern = re.compile(r"^mjpeg(:[:a-z0-9A-Z\-]+)?$", re.IGNORECASE)
        if mjpeg_pattern.match(path):
            mjpeg_query = mjpeg_pattern.match(path).group(1) 
            mjpeg_args = []
            if mjpeg_query is not None:
                mjpeg_args = mjpeg_query[1:].split(":")
            n_mjpeg_args = len(mjpeg_args)
            if n_mjpeg_args == 0:
                host, port = "localhost", 8080
            elif n_mjpeg_args == 1:
                host, port = "localhost", int(mjpeg_args[0])
            elif n_mjpeg_args == 2:
                host, port = mjpeg_args[1], int(mjpeg_args[0])
            else:
                raise ValueError(f"Invalid number of MJPEG arguments: {n_mjpeg_args}")
            return MjpegOutput(host, port, width, height, framerate)
        return FFmpegVideoOutput(path, width, height, framerate, vcodec,
                                 execute, replace, safe)


class FFmpegVideoOutput(VideoOutput):

    def __init__(self, path: str, width: int, height: int, framerate: int,
                 vcodec: str = "h264", execute: bool = False,
                 replace: bool = False, safe: bool = False):
        VideoOutput.__init__(self, width, height)
        self.path = path if replace else find_unique_path(path)
        self.framerate = framerate
        self.vcodec = vcodec
        self.execute = execute
        self.safe = safe
        self.process = None

    def __enter__(self):
        dirname = os.path.dirname(self.path)
        if not os.path.isdir(dirname) and dirname != "":
            os.makedirs(dirname)
        if self.safe:
            append_history(self.path)
        self.process = subprocess.Popen([
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec","rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", f"{self.framerate}",
            "-i", "-",
            "-r", f"{self.framerate}",
            "-pix_fmt", "yuv420p",
            "-an",
            "-vcodec", self.vcodec,
            self.path,
            "-y"
        ], stdin=subprocess.PIPE)
        return self

    def feed(self, frame: numpy.ndarray):
        self.process.stdin.write(frame.astype(numpy.uint8).tobytes())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.process.stdin.close()
        self.process.wait()
        if self.execute:
            startfile(self.path)


class CvVideoOutput(VideoOutput):

    WINDOW_NAME = "TransFlow"

    def __enter__(self):
        cv2.namedWindow(CvVideoOutput.WINDOW_NAME, cv2.WINDOW_NORMAL)
        return self

    def feed(self, frame: numpy.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(CvVideoOutput.WINDOW_NAME, frame)
        cv2.waitKey(1)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        cv2.destroyAllWindows()


class MjpegOutput(VideoOutput):

    def __init__(self, host: str, port: int, width: int, height: int,
                 framerate: int, quality: int = 50):
        VideoOutput.__init__(self, width, height)
        self.host = host
        self.port = port
        self.framerate = framerate
        self.quality = quality
        self.stream = None
        self.server = None
    
    def __enter__(self):
        self.stream = MjpegStream("transflow", size=(self.width, self.height), quality=self.quality, fps=self.framerate)
        self.server = MjpegServer(self.host, self.port)
        self.server.add_stream(self.stream)
        self.server.start()
        return self
    
    def feed(self, frame: numpy.ndarray):
        self.stream.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.server.stop()


class ZipOutput:

    def __init__(self, path: str, replace: bool = False):
        import zipfile
        self.path = path if replace else find_unique_path(path)
        if os.path.isfile(self.path):
            os.remove(self.path)
        self.archive = zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED)

    def write_meta(self, data: dict):
        if not data:
            return
        import json
        with self.archive.open("meta.json", "w") as file:
            file.write(json.dumps(data).encode())

    def write_object(self, filename: str, obj: object):
        import pickle
        with self.archive.open(filename, "w") as file:
            pickle.dump(obj, file)

    def close(self):
        self.archive.close()


class NumpyOutput(ZipOutput):

    def __init__(self, path: str, replace: bool = False):
        ZipOutput.__init__(self, path, replace)
        self.index = 0

    def write_array(self, array: numpy.ndarray):
        with self.archive.open(f"{self.index:09d}.npy", "w") as file:
            numpy.save(file, array)
        self.index += 1


def render1d(arr: numpy.ndarray, scale: float = 1,
             colors: tuple[str] | None = None, binary: bool = False
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
              colors: tuple[str] | None = None)-> numpy.ndarray:
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
