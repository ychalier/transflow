import multiprocessing

import numpy

from ..types import Pixmap, BoolMask


class EndOfPixmap(StopIteration):
    pass


class PixmapSourceInterface:

    def __init__(self, queue: multiprocessing.Queue, introduction_mask: BoolMask):
        self.queue = queue
        self.image: Pixmap | None = None
        self.counter: int = -1
        self.introduction_mask: BoolMask = introduction_mask

    def get(self) -> Pixmap:
        assert self.image is not None
        return self.image

    def next(self, timeout: float = 1) -> Pixmap:
        image = self.queue.get(timeout=timeout)
        if image is None:
            raise EndOfPixmap
        assert isinstance(image, numpy.ndarray)
        assert len(image.shape) == 3
        assert image.dtype == numpy.uint8
        self.image = image
        self.counter += 1
        return self.image

    @property
    def frame_number(self) -> int:
        return self.counter
