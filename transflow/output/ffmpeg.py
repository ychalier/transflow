import logging
import os
import subprocess

import numpy

from .video_output import VideoOutput
from ..utils import find_unique_path, startfile


logger = logging.getLogger(__name__)


def append_history(output_path): # TODO: remove this?
    with open("history.log", "a", encoding="utf8") as file:
        file.write(f"└► {os.path.realpath(output_path)}\n")


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
        logger.debug("Started FFmpeg output to %s", self.path)
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
        if self.process is None or self.process.stdin is None:
            raise ValueError("Process not initialized")
        self.process.stdin.write(frame.astype(numpy.uint8).tobytes())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        logger.debug("Interrupting FFmpeg process")
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
        if self.process is not None:
            self.process.wait()
        if self.execute:
            startfile(self.path)
