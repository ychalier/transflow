import warnings

import cv2
import numpy

from .source import PixmapSource


class CvPixmapSource(PixmapSource):

    def __init__(self, path: str, seek: int | None = None,
                 seek_time: float | None = None,
                 alteration_path: str | None = None, repeat: int = 1):
        PixmapSource.__init__(self, alteration_path)
        self.path = path
        self.capture = None
        self.seek = seek
        self.seek_time = seek_time
        self.repeat = repeat
        self.loop_index = 1

    def rewind(self):
        if self.capture is None:
            raise ValueError("Capture not initialized")
        self.capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        if self.seek is not None:
            for _ in range(self.seek):
                self.capture.read()

    def __enter__(self):
        self.setup()
        self.capture = cv2.VideoCapture(self.path)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = round(self.capture.get(cv2.CAP_PROP_FPS))
        frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.repeat > 0 and frame_count is not None and int(frame_count) > 0:
            self.length = int(frame_count) * self.repeat
        if self.seek_time is not None:
            self.seek = int(self.seek_time * self.framerate)
            if self.length is not None:
                self.length -= self.seek * self.repeat
        self.rewind()
        return self

    def __next__(self) -> numpy.ndarray:
        assert self.capture is not None
        if not self.capture.isOpened():
            warnings.warn("Attempt to read frame from pixmap capture, which was not opened")
            raise StopIteration
        while True:
            success, frame = self.capture.read()
            if success and frame is not None:
                break
            if self.repeat == 0 or self.loop_index < self.repeat:
                self.loop_index += 1
                self.rewind()
                continue
            raise StopIteration
        return self.alter(numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.capture is not None:
            self.capture.release()
