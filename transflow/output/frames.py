import logging
import os
import pathlib

import cv2
import numpy

from .video_output import VideoOutput
from ..utils import startfile


logger = logging.getLogger(__name__)


class FramesVideoOutput(VideoOutput):

    def __init__(self, template: str, width: int, height: int, execute: bool = False):
        VideoOutput.__init__(self, width, height)
        self.template = template
        self.directory = pathlib.Path(self.template).parent
        self.execute = execute
        self.counter = 0

    def __enter__(self):
        if not os.path.isdir(self.directory) and self.directory != "":
            os.makedirs(self.directory)
        logger.debug("Started frames output to %s", self.directory)
        self.counter = 0
        return self

    def feed(self, frame: numpy.ndarray):
        cv2.imwrite(self.template % self.counter, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.counter += 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.execute:
            startfile(self.directory.as_posix())
