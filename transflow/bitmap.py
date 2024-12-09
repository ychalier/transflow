import os
import random
import re
import typing

import cv2
import numpy

from .utils import parse_hex_color


class BitmapSource:

    def __init__(self, alteration_path: str | None):
        self.alteration_path: str | None = alteration_path
        self.width: int = None
        self.height: int = None
        self.framerate: int | None = None
        self.alteration: tuple[list[int], list[int]] | None = None

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
    def from_args(cls, path: str, size: tuple[int, int] | None = None,
                  seek: int | None = None, seed: int | None = None,
                  seek_time: float | None = None,
                  alteration_path: str | None = None):
        ext = os.path.splitext(path)[1]
        stillm = re.match(r"(color|noise|bwnoise|cnoise|gradient|#?[0-9a-f]{6})(:\d+:\d+)?",
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
                    return GradientNoiseBitmapSource(width, height, seed, alteration_path)
                case _:
                    raise ValueError(f"Unknown bitmap source '{stillm.group(1)}'")
        elif os.path.isfile(path) and ext in {".jpg", ".jpeg", ".png"}:
            return ImageBitmapSource(path, alteration_path)
        else:
            return CvBitmapSource(path, seek, seek_time, alteration_path)


class StillBitmapSource(BitmapSource):

    def __init__(self, width: int | None = None, height: int | None = None,
                 seed: int | None = None, alteration_path: str | None = None):
        BitmapSource.__init__(self, alteration_path)
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


class GradientNoiseBitmapSource(StillBitmapSource):

    CHAIKIN_ITERATIONS = 3
    BLOB_COUNT = 3
    SX = 300
    SY = 300
    MIN_POINTS = 3
    MAX_POINTS = 6

    @staticmethod
    def fig2rgb_array(fig):
        """adapted from: https://stackoverflow.com/questions/21939658/"""
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        return numpy.frombuffer(buf, dtype=numpy.uint8).reshape(nrows, ncols, 3)

    def _generate_blob(self):
        import scipy.spatial
        cx = random.random() * self.width
        cy = random.random() * self.height
        points = []
        for _ in range(random.randint(self.MIN_POINTS, self.MAX_POINTS)):
            px = random.normalvariate(cx, self.SX)
            py = random.normalvariate(cy, self.SY)
            points.append((px, py))
        array = numpy.array(points)
        hull = scipy.spatial.ConvexHull(array)
        ipoints = array[hull.vertices].tolist()
        ipoints.append(points[hull.vertices[0]])
        for _ in range(self.CHAIKIN_ITERATIONS):
            jpoints = []
            for a, b in zip(ipoints, ipoints[1:]):
                ab = [b[0] - a[0], b[1] - a[1]]
                jpoints.append((a[0] + .25 * ab[0], a[1] + .25 * ab[1]))
                jpoints.append((a[0] + .75 * ab[0], a[1] + .75 * ab[1]))
            ipoints = jpoints[:]
        kpoints = [ipoints[-1], points[hull.vertices[0]], ipoints[0]]
        for _ in range(self.CHAIKIN_ITERATIONS):
            jpoints = [kpoints[0]]
            for a, b in zip(kpoints, kpoints[1:]):
                ab = [b[0] - a[0], b[1] - a[1]]
                jpoints.append((a[0] + .25 * ab[0], a[1] + .25 * ab[1]))
                jpoints.append((a[0] + .75 * ab[0], a[1] + .75 * ab[1]))
            kpoints = jpoints[:] + [kpoints[-1]]
        return ipoints + kpoints

    def _init_array(self):
        import matplotlib.pyplot, scipy.ndimage
        numpy.random.seed(self.seed)
        random.seed(self.seed)
        array = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        array[:,:] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(self.BLOB_COUNT):
            blob = self._generate_blob()
            fig = matplotlib.pyplot.figure(figsize=(self.width/72, self.height/72), dpi=72)
            ax = fig.gca()
            ax.set_axis_off()
            matplotlib.pyplot.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            matplotlib.pyplot.margins(0, 0)
            ax.set_xlim(0, self.width - 1)
            ax.set_ylim(0, self.height - 1)
            ax.imshow(array, aspect="equal", origin="lower")
            ax.fill(
                [p[0] for p in blob], [p[1] for p in blob],
                color=(random.random(), random.random(), random.random(), 1))
            array = scipy.ndimage.gaussian_filter(self.fig2rgb_array(fig), sigma=(31, 31, 0))
            matplotlib.pyplot.close()
        noise = numpy.sqrt(numpy.random.random((self.height, self.width, 1)))
        return (noise * array).astype(numpy.uint8)


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


class CvBitmapSource(BitmapSource):

    def __init__(self, path: str, seek: int | None = None,
                 seek_time: float | None = None,
                 alteration_path: str | None = None):
        BitmapSource.__init__(self, alteration_path)
        self.path = path
        self.capture = None
        self.seek = seek
        self.seek_time = seek_time

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.path)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.capture.get(cv2.CAP_PROP_FPS)
        if self.seek_time is not None:
            self.seek = int(self.seek_time * self.framerate)
        if self.seek is not None:
            for _ in range(self.seek):
                self.capture.read()
        self.setup()
        return self

    def __next__(self) -> numpy.ndarray:
        assert self.capture is not None
        if not self.capture.isOpened():
            raise StopIteration
        success, frame = self.capture.read()
        if not success or frame is None:
            raise StopIteration
        return self.alter(numpy.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.capture.release()
