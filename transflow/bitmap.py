import os
import random
import re
import typing
import warnings

import cv2
import numpy

from .utils import parse_hex_color


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ico", ".tiff"}


class BitmapSource:

    def __init__(self, alteration_path: str | None, length: int | None = None):
        self.alteration_path: str | None = alteration_path
        self.width: int = None
        self.height: int = None
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
        image = numpy.array(PIL.Image.open(self.alteration_path))
        inds = []
        vals = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 3] == 0:
                    continue
                k = (i * self.width + j) * 3
                inds += [k, k + 1, k + 2]
                vals += image[i, j, :3].tolist()
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
                    return ColorBitmapSource(width, height, seed=seed, alteration_path=alteration_path)
                case _ if re.match(r"#?[0-9a-f]{6}", stillm.group(1)):
                    return ColorBitmapSource(width, height, stillm.group(1), seed=seed, alteration_path=alteration_path)
                case "noise":
                    return NoiseBitmapSource(width, height, seed, alteration_path)
                case "bwnoise":
                    return BwNoiseBitmapSource(width, height, seed, alteration_path)
                case "cnoise":
                    return ColoredNoiseBitmapSource(width, height, seed, alteration_path)
                case "gradient":
                    return GradientBitmapSource(width, height, seed)
                case "first":
                    assert flow_path is not None
                    return VideoStillBitmapSource(flow_path, alteration_path)
                case _:
                    raise ValueError(f"Unknown bitmap source '{stillm.group(1)}'")
        elif os.path.isfile(path) and ext.lower() in IMAGE_EXTS:
            return ImageBitmapSource(path, alteration_path)
        else:
            return CvBitmapSource(path, seek, seek_time, alteration_path, repeat)


class StillBitmapSource(BitmapSource):

    def __init__(self, width: int | None = None, height: int | None = None,
                 seed: int | None = None, alteration_path: str | None = None):
        BitmapSource.__init__(self, alteration_path, length=None)
        self.width = width
        self.height = height
        self.seed = seed
        self.array: numpy.ndarray | None = None

    def _init_array(self) -> numpy.ndarray:
        raise NotImplementedError()

    def __enter__(self):
        self.array = self._init_array()
        self.width = self.array.shape[1]
        self.height = self.array.shape[0]
        self.setup()
        return self

    def __next__(self) -> numpy.ndarray:
        assert self.array is not None
        return self.alter(self.array.copy())


class ColorBitmapSource(StillBitmapSource):

    def __init__(self, width: int, height: int, bitmap_color: str | None = None,
                 seed: int | None = None, alteration_path: str | None = None):
        StillBitmapSource.__init__(self, width, height, seed, alteration_path)
        self.bitmap_color = bitmap_color

    def _init_array(self):
        numpy.random.seed(self.seed)
        if self.bitmap_color is None:
            color = list(numpy.random.randint(0, 256, size=(3), dtype=numpy.uint8))
        else:
            color = parse_hex_color(self.bitmap_color)
        array = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        array[:,:,:] = color
        return array


class NoiseBitmapSource(StillBitmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        return numpy.repeat(
            numpy.random.randint(0, 256, size=(self.height, self.width, 1), dtype=numpy.uint8),
            3, axis=2)


class BwNoiseBitmapSource(StillBitmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        return numpy.repeat(numpy.random.choice([0, 255], size=(self.height, self.width, 1)),
                            3, axis=2).astype(numpy.uint8)


class ColoredNoiseBitmapSource(StillBitmapSource):

    def _init_array(self):
        numpy.random.seed(self.seed)
        return numpy.random.randint(0, 256, size=(self.height, self.width, 3), dtype=numpy.uint8)


class GradientBitmapSource(StillBitmapSource):

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
    
    def evaluate(self, tree: tuple, i: int, j: int) -> tuple[float, float, float]:
        nt, a, b, c = tree
        if nt == self.NODE_TRIPLE:
            return (
                self.evaluate(a, i, j)[0],
                self.evaluate(b, i, j)[1],
                self.evaluate(c, i, j)[2])
        if nt == self.NODE_MIX:
            out = [0, 0, 0]
            evals = [
                self.evaluate(a, i, j),
                self.evaluate(b, i, j),
                self.evaluate(c, i, j)]
            for k in range(3):
                w = (1 + evals[0][k]) / 2
                out[k] = (1 - w) * evals[1][k] + w * evals[2][k]
            return out
        if nt == self.NODE_RGB:
            return (a, b, c)
        if nt == self.NODE_I:
            z = 2 * (i / (self.height - 1)) - 1
            return (z, z, z)
        if nt == self.NODE_J:
            z = 2 * (j / (self.width - 1)) - 1
            return (z, z, z)
        return NotImplementedError(f"Unknown node type {nt}")

    def _init_array(self):
        random.seed(self.seed)
        tree = self.generate(self.NODE_TRIPLE, 5)
        array = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        for i in range(self.height):
            for j in range(self.width):
                r, g, b = self.evaluate(tree, i, j)
                array[i, j, 0] = 255 * (r + 1) / 2
                array[i, j, 1] = 255 * (g + 1) / 2
                array[i, j, 2] = 255 * (b + 1) / 2
        return array.astype(numpy.uint8)


class ImageBitmapSource(StillBitmapSource):

    def __init__(self, path: str, alteration_path: str | None = None):
        StillBitmapSource.__init__(self, alteration_path=alteration_path)
        self.path = path

    def _init_array(self):
        import PIL.Image
        image = PIL.Image.open(self.path)
        array = numpy.array(image)[:,:,:3]
        image.close()
        return array


class VideoStillBitmapSource(ImageBitmapSource):
    
    def _init_array(self):
        capture = cv2.VideoCapture(self.path)
        success, frame = capture.read()
        assert success, "Could not open video for still bitmap source"
        array = numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return array


class CvBitmapSource(BitmapSource):

    def __init__(self, path: str, seek: int | None = None,
                 seek_time: float | None = None,
                 alteration_path: str | None = None, repeat: int = 1):
        BitmapSource.__init__(self, alteration_path)
        self.path = path
        self.capture = None
        self.seek = seek
        self.seek_time = seek_time
        self.repeat = repeat
        self.loop_index = 1
    
    def rewind(self):
        self.capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        if self.seek is not None:
            for _ in range(self.seek):
                self.capture.read()

    def __enter__(self):
        self.setup()
        self.capture = cv2.VideoCapture(self.path)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.capture.get(cv2.CAP_PROP_FPS)
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
            warnings.warn("Attempt to read frame from bitmap capture, which was not opened")
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
        self.capture.release()

