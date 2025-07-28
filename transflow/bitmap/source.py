import logging
import os
import re
import typing
from typing import cast

import numpy


logger = logging.getLogger(__name__)


class BitmapSource:

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

    def __next__(self) -> numpy.ndarray:
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

    def alter(self, array: numpy.ndarray) -> numpy.ndarray:
        if self.alteration is None:
            return array
        numpy.put(array, self.alteration[0], self.alteration[1])
        return array

    @classmethod
    def from_args(cls,
            path: str,
            size: tuple[int, int] | None = None,
            seek: int | None = None,
            seed: int | None = None,
            seek_time: float | None = None,
            alteration_path: str | None = None,
            repeat: int = 1,
            flow_path: str | None = None):
        ext = os.path.splitext(path)[1]
        stillm = re.match(r"(color|noise|bwnoise|cnoise|gradient|first|#?[0-9a-f]{6})(:\d+:\d+)?",
                          path.lower().strip())
        if stillm is not None:
            if stillm.group(2) is not None:
                width, height = tuple(map(int, stillm.group(2).split(":")[1:]))
            elif size is not None:
                width, height = size
            else:
                raise ValueError(f"Please specify a resolution with {stillm.group(1)}:width:height")
            match stillm.group(1):
                case "color":
                    from .still import ColorBitmapSource
                    return ColorBitmapSource(width, height, seed=seed, alteration_path=alteration_path)
                case _ if re.match(r"#?[0-9a-f]{6}", stillm.group(1)):
                    from .still import ColorBitmapSource
                    return ColorBitmapSource(width, height, stillm.group(1), seed=seed, alteration_path=alteration_path)
                case "noise":
                    from .still import NoiseBitmapSource
                    return NoiseBitmapSource(width, height, seed, alteration_path)
                case "bwnoise":
                    from .still import BwNoiseBitmapSource
                    return BwNoiseBitmapSource(width, height, seed, alteration_path)
                case "cnoise":
                    from .still import ColoredNoiseBitmapSource
                    return ColoredNoiseBitmapSource(width, height, seed, alteration_path)
                case "gradient":
                    from .still import GradientBitmapSource
                    return GradientBitmapSource(width, height, seed)
                case "first":
                    from .still import VideoStillBitmapSource
                    assert flow_path is not None
                    return VideoStillBitmapSource(flow_path, alteration_path)
                case _:
                    raise ValueError(f"Unknown bitmap source '{stillm.group(1)}'")
        elif os.path.isfile(path) and ext.lower() in BitmapSource.IMAGE_EXTS:
            from .still import ImageBitmapSource
            return ImageBitmapSource(path, alteration_path)
        else:
            from .cv import CvBitmapSource
            return CvBitmapSource(path, seek, seek_time, alteration_path, repeat)

