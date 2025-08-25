import logging
import os
import re
import typing
from typing import cast

import numpy

from ..types import Pixmap


logger = logging.getLogger(__name__)


class PixmapSource:

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ico", ".tiff"}

    def __init__(self, alteration_path: str | None, length: int | None = None):
        logger.debug("Initializing '%s'", self.__class__.__name__)
        self.alteration_path: str | None = alteration_path
        self.width: int | None = None
        self.height: int | None = None
        self.framerate: int | None = None
        self.alteration: tuple[list[int], list[int]] | None = None
        self.length: int | None = length

    def __enter__(self) -> typing.Self:
        return self

    def __next__(self) -> Pixmap:
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def load_alteration(self):
        if self.alteration_path is None:
            return
        import PIL.Image
        logger.debug("Loading alteration at %s", self.alteration_path)
        image = numpy.array(PIL.Image.open(self.alteration_path))
        while image.shape[2] < 4:
            appendee = numpy.ones((*image.shape[:2],1), dtype=numpy.uint8)
            image = numpy.append(image, appendee, 2)
        inds = []
        vals = []
        if self.width is None:
            raise ValueError("Width not initialized")
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 3] == 0:
                    continue
                k = (i * self.width + j) * 3
                inds += [k, k + 1, k + 2]
                vals += cast(list[int], image[i, j, :3].tolist())
        self.alteration = (inds, vals)

    def setup(self):
        self.load_alteration()

    def alter(self, array: Pixmap) -> Pixmap:
        if self.alteration is None:
            return array
        numpy.put(array, self.alteration[0], self.alteration[1])
        return array

    @classmethod
    def from_args(cls,
            path: str,
            size: tuple[int, int],
            seek: int | None = None,
            seed: int | None = None,
            seek_time: float | None = None,
            alteration_path: str | None = None,
            repeat: int = 1,
            flow_path: str | None = None):
        ext = os.path.splitext(path)[1]
        still_match = re.match(
            r"^(color:[a-z0-9\(\)#, ]+|color|#?[0-9a-f]{6}|noise|bwnoise|cnoise|gradient|first)$",
            path.lower().strip())
        if still_match is not None:
            width, height = size
            still_class = still_match.group(1)
            if still_class == "color":
                from .still import ColorPixmapSource
                return ColorPixmapSource(width, height, seed=seed, alteration_path=alteration_path)
            elif still_class.startswith("color:"):
                from .still import ColorPixmapSource
                return ColorPixmapSource(width, height, still_class.split(":", 1)[1], seed=seed, alteration_path=alteration_path)
            elif re.match(r"#?[0-9a-f]{6}", still_class):
                from .still import ColorPixmapSource
                return ColorPixmapSource(width, height, still_class, seed=seed, alteration_path=alteration_path)
            elif still_class == "noise":
                from .still import NoisePixmapSource
                return NoisePixmapSource(width, height, seed, alteration_path)
            elif still_class == "bwnoise":
                from .still import BwNoisePixmapSource
                return BwNoisePixmapSource(width, height, seed, alteration_path)
            elif still_class == "cnoise":
                from .still import ColoredNoisePixmapSource
                return ColoredNoisePixmapSource(width, height, seed, alteration_path)
            elif still_class == "gradient":
                from .still import GradientPixmapSource
                return GradientPixmapSource(width, height, seed)
            elif still_class == "first":
                from .still import VideoStillPixmapSource
                assert flow_path is not None
                return VideoStillPixmapSource(flow_path, alteration_path)
            else:
                raise ValueError(f"Unknown pixmap source '{still_match.group(1)}'")
        elif os.path.isfile(path) and ext.lower() in PixmapSource.IMAGE_EXTS:
            from .still import ImagePixmapSource
            return ImagePixmapSource(path, alteration_path)
        else:
            from .cv import CvPixmapSource
            return CvPixmapSource(path, seek, seek_time, alteration_path, repeat)

