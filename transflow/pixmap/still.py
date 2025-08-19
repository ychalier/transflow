import random
from typing import cast

import cv2
import numpy

from .source import PixmapSource
from ..utils import parse_hex_color
from ..types import Rgb, Pixmap


class StillPixmapSource(PixmapSource):

    def __init__(self, width: int | None = None, height: int | None = None,
                 seed: int | None = None, alteration_path: str | None = None):
        PixmapSource.__init__(self, alteration_path, length=None)
        self.width = width
        self.height = height
        self.seed = seed
        self.array: Pixmap | None = None

    def _init_array(self) -> Pixmap:
        raise NotImplementedError()

    def __enter__(self):
        self.array = self._init_array()
        self.width = self.array.shape[1]
        self.height = self.array.shape[0]
        self.setup()
        return self

    def __next__(self) -> Pixmap:
        assert self.array is not None
        return self.alter(self.array.copy())


class ColorPixmapSource(StillPixmapSource):

    def __init__(self, width: int, height: int, color: str | None = None,
                 seed: int | None = None, alteration_path: str | None = None):
        StillPixmapSource.__init__(self, width, height, seed, alteration_path)
        self.color = color

    def _init_array(self):
        numpy.random.seed(self.seed)
        if self.color is None:
            color = list(numpy.random.randint(0, 256, size=(3), dtype=numpy.uint8))
        else:
            color = parse_hex_color(self.color)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        array = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        array[:,:,:] = color
        return cast(Rgb, array)


class NoisePixmapSource(StillPixmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        return cast(Rgb, numpy.repeat(numpy.random.randint(0, 256, size=(self.height, self.width, 1), dtype=numpy.uint8), 3, axis=2))


class BwNoisePixmapSource(StillPixmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        return cast(Rgb, numpy.repeat(numpy.random.choice([0, 255], size=(self.height, self.width, 1)), 3, axis=2).astype(numpy.uint8))


class ColoredNoisePixmapSource(StillPixmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        return cast(Rgb, numpy.random.randint(0, 256, size=(self.height, self.width, 3), dtype=numpy.uint8))


class GradientPixmapSource(StillPixmapSource):

    NODE_I = 0
    NODE_J = 1
    NODE_RGB = 2
    NODE_MIX = 3
    NODE_TRIPLE = 4
    NODE_Z = 5
    NODE_B = 6

    def generate(self, node_type: int, depth: int) -> tuple:
        if depth <= 0 and node_type != self.NODE_Z:
            return self.generate(self.NODE_Z, 0)
        if node_type in [self.NODE_TRIPLE, self.NODE_MIX]:
            return (
                node_type,
                self.generate(self.NODE_B, depth - 1),
                self.generate(self.NODE_B, depth - 1),
                self.generate(self.NODE_B, depth - 1)
            )
        elif node_type == self.NODE_B:
            if random.random() < .25:
                return self.generate(self.NODE_Z, depth - 1)
            return self.generate(self.NODE_MIX, depth - 1)
        elif node_type == self.NODE_Z:
            x = random.random()
            if x < .333:
                return (self.NODE_I, None, None, None)
            elif x < .666:
                return (self.NODE_J, None, None, None)
            return (
                self.NODE_RGB,
                random.random() * 2 - 1,
                random.random() * 2 - 1,
                random.random() * 2 - 1)
        raise ValueError(f"Unkown node type {node_type}")

    def evaluate(self, tree: tuple, i: int, j: int) -> tuple[float, float, float]:
        nt, a, b, c = tree
        if nt == self.NODE_TRIPLE:
            return (
                self.evaluate(a, i, j)[0],
                self.evaluate(b, i, j)[1],
                self.evaluate(c, i, j)[2])
        if nt == self.NODE_MIX:
            out: list[float] = [0, 0, 0]
            evals = [
                self.evaluate(a, i, j),
                self.evaluate(b, i, j),
                self.evaluate(c, i, j)
            ]
            for k in range(3):
                w = (1 + evals[0][k]) / 2
                out[k] = (1 - w) * evals[1][k] + w * evals[2][k]
            return (out[0], out[1], out[2])
        if nt == self.NODE_RGB:
            return (a, b, c)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        if nt == self.NODE_I:
            z = 2 * (i / (self.height - 1)) - 1
            return (z, z, z)
        if nt == self.NODE_J:
            z = 2 * (j / (self.width - 1)) - 1
            return (z, z, z)
        raise NotImplementedError(f"Unknown node type {nt}")

    def _init_array(self):
        random.seed(self.seed)
        tree = self.generate(self.NODE_TRIPLE, 5)
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        array = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        for i in range(self.height):
            for j in range(self.width):
                r, g, b = self.evaluate(tree, i, j)
                array[i, j, 0] = 255 * (r + 1) / 2
                array[i, j, 1] = 255 * (g + 1) / 2
                array[i, j, 2] = 255 * (b + 1) / 2
        return cast(Rgb, array.astype(numpy.uint8))


class ImagePixmapSource(StillPixmapSource):

    def __init__(self, path: str, alteration_path: str | None = None):
        StillPixmapSource.__init__(self, alteration_path=alteration_path)
        self.path = path

    def _init_array(self):
        import PIL.Image
        image = PIL.Image.open(self.path)
        array = numpy.array(image)[:,:,:]
        image.close()
        assert array.shape[2] == 3 or array.shape[2] == 4, f"Pixmap image has unsupported dimension: {array.shape}"
        return cast(Pixmap, array)


class VideoStillPixmapSource(ImagePixmapSource):

    def _init_array(self):
        capture = cv2.VideoCapture(self.path)
        success, frame = capture.read()
        assert success, "Could not open video for still bitmap source"
        array = numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return cast(Rgb, array)
