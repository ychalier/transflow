import logging
import os
import pathlib

import cv2

from .video_output import VideoOutput
from ..utils import startfile
from ..types import Rgb


logger = logging.getLogger(__name__)


class FramesVideoOutput(VideoOutput):

    def __init__(self, template: str, width: int, height: int, initial_counter: int = 0, execute: bool = False):
        VideoOutput.__init__(self, width, height)
        self.template = template
        self.directory = pathlib.Path(self.template).parent
        self.execute = execute
        self.counter = initial_counter

    def __enter__(self):
        if not os.path.isdir(self.directory) and self.directory != "":
            os.makedirs(self.directory)
        logger.debug("Started frames output to %s", self.directory)
        return self

    def feed(self, frame: Rgb):
        cv2.imwrite(self.template % self.counter, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.counter += 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.execute:
            startfile(self.directory.as_posix())
