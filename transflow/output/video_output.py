import re

import numpy


class VideoOutput:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __enter__(self):
        return self

    def feed(self, frame: numpy.ndarray):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    @classmethod
    def from_args(cls, path: str | None, width: int, height: int,
                  framerate: int | None = None, vcodec: str = "h264",
                  execute: bool = False, replace: bool = False,
                  safe: bool = False):
        if path is None:
            from .cv import CvVideoOutput
            return CvVideoOutput(width, height)
        if framerate is None:
            framerate = 30
        mjpeg_pattern = re.compile(r"^mjpeg(:[:a-z0-9A-Z\-]+)?$", re.IGNORECASE)
        m = mjpeg_pattern.match(path)
        if m:
            mjpeg_query = m.group(1) 
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
            from .mjpeg import MjpegOutput
            return MjpegOutput(host, port, width, height, framerate)
        from .ffmpeg import FFmpegVideoOutput
        return FFmpegVideoOutput(path, width, height, framerate, vcodec,
                                 execute, replace, safe)
