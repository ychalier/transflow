import unittest

import numpy

import transflow.bitmap


class TestBitmapSource(unittest.TestCase):

    VIDEO_PATH = "assets/River.mp4"
    MASK_PATH = "assets/Mask.png"
    WIDTH = 854
    HEIGHT = 480
    FPS = 50
    LENGTH = 1500

    def _test_bs(self, bs: transflow.bitmap.BitmapSource):
        with bs:
            self.assertEqual(bs.width, self.WIDTH)
            self.assertEqual(bs.height, self.HEIGHT)
            bitmap = next(bs)
            self.assertIsInstance(bitmap, numpy.ndarray)
            self.assertEqual(bitmap.shape, (self.HEIGHT, self.WIDTH, 3))
            self.assertEqual(bitmap.dtype, numpy.uint8)
        return bitmap

    def test_color_random(self):
        bs = transflow.bitmap.BitmapSource.from_args("color", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.ColorBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_color_specific(self):
        bs = transflow.bitmap.BitmapSource.from_args("cff010", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.ColorBitmapSource)
        with bs:
            bitmap = self._test_bs(bs)
            self.assertEqual(bitmap[0,0,0], 207)
            self.assertEqual(bitmap[0,0,1], 240)
            self.assertEqual(bitmap[0,0,2], 16)

    def test_noise(self):
        bs = transflow.bitmap.BitmapSource.from_args("noise", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.NoiseBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_bwnoise(self):
        bs = transflow.bitmap.BitmapSource.from_args("bwnoise", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.BwNoiseBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_cnoise(self):
        bs = transflow.bitmap.BitmapSource.from_args("cnoise", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.ColoredNoiseBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_gradient(self):
        bs = transflow.bitmap.BitmapSource.from_args("gradient", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.GradientBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_image(self):
        bs = transflow.bitmap.BitmapSource.from_args("assets/Deer.jpg", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.ImageBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_video_still(self):
        bs = transflow.bitmap.BitmapSource.from_args("first", (self.WIDTH, self.HEIGHT), flow_path="assets/River.mp4")
        self.assertIsInstance(bs, transflow.bitmap.VideoStillBitmapSource)
        with bs:
            self._test_bs(bs)

    def test_video(self):
        bs = transflow.bitmap.BitmapSource.from_args("assets/River.mp4", (self.WIDTH, self.HEIGHT))
        self.assertIsInstance(bs, transflow.bitmap.CvBitmapSource)
        with bs:
            self._test_bs(bs)
            self.assertEqual(bs.length, 1500)
            if isinstance(bs, transflow.bitmap.CvBitmapSource):
                bs.rewind()

    def test_alteration(self):
        bs = transflow.bitmap.BitmapSource.from_args("cnoise", (self.WIDTH, self.HEIGHT), alteration_path="assets/Mask.png")
        import PIL.Image
        image = PIL.Image.open("assets/Mask.png")
        array = numpy.array(image)[:,:,:3]
        with bs:
            bitmap = self._test_bs(bs)
        self.assertEqual(numpy.sum(array - bitmap), 0)
