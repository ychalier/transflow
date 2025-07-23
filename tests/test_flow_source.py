import os
import unittest

import numpy

import transflow.flow
import transflow.flow.sources.av
import transflow.flow.sources.cv


class TestFlowSource(unittest.TestCase):

    VIDEO_PATH = "assets/River.mp4"
    MASK_PATH = "assets/Mask.png"
    WIDTH = 854
    HEIGHT = 480
    FPS = 50
    LENGTH = 1500

    def _test_fs(self, fs: transflow.flow.FlowSource, length: int | None = None):
        if length is None:
            length = self.LENGTH - 1
        with fs:
            self.assertEqual(fs.width, self.WIDTH)
            self.assertEqual(fs.height, self.HEIGHT)
            self.assertEqual(fs.framerate, self.FPS)
            self.assertEqual(fs.length, length)
            flow = next(fs)
            self.assertIsInstance(flow, numpy.ndarray)
            self.assertEqual(flow.shape, (self.HEIGHT, self.WIDTH, 2))
            self.assertEqual(flow.dtype, numpy.float32)

    def test_cv_forward(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH,
            direction=transflow.flow.FlowDirection.FORWARD)
        self.assertIsInstance(fs, transflow.flow.sources.cv.CvFlowSource)
        self._test_fs(fs)
    
    def test_cv_backward(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH,
            direction=transflow.flow.FlowDirection.BACKWARD)
        self.assertIsInstance(fs, transflow.flow.sources.cv.CvFlowSource)
        self._test_fs(fs)
    
    def test_av(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, use_mvs=True)
        self.assertIsInstance(fs, transflow.flow.sources.av.AvFlowSource)
        self._test_fs(fs)
    
    def test_cv_av_timings(self):
        for use_mvs in [True, False]:
            seek_time = 1
            duration_time = 3
            repeats = 3
            fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, use_mvs=use_mvs, repeat=repeats, seek_time=seek_time)
            self._test_fs(fs, (self.LENGTH - self.FPS * seek_time - 1) * repeats)
            fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, use_mvs=use_mvs, repeat=repeats, duration_time=duration_time)
            self._test_fs(fs, self.FPS * duration_time * repeats)
    
    def test_cv_lock(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, lock_expr="t >= 1", lock_mode=transflow.flow.LockMode.SKIP)
        self._test_fs(fs)
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, lock_expr="(1,1)", lock_mode=transflow.flow.LockMode.STAY)
        self._test_fs(fs, self.LENGTH + self.FPS - 1)

    def test_mask(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, mask_path=self.MASK_PATH)
        self._test_fs(fs)
        self.assertIsNotNone(fs.mask)
        if fs.mask is not None:
            self.assertEqual(fs.mask.shape, (self.HEIGHT, self.WIDTH, 1))

    def test_blurred(self):
        self._test_fs(transflow.flow.FlowSource.from_args(self.VIDEO_PATH, cv_config="configs/blurred.json"))
    
    def test_horn_schunck(self):
        self._test_fs(transflow.flow.FlowSource.from_args(self.VIDEO_PATH, cv_config="configs/horn-schunck.json"))
    
    def test_lukas_kanade(self):
        self._test_fs(transflow.flow.FlowSource.from_args(self.VIDEO_PATH, cv_config="configs/lukas-kanade.json"))

    # def test_liteflownet(self):
    #     try:
    #         import cupy
    #     except ImportError:
    #         return
    #     self._test_fs(transflow.flow.FlowSource.from_args(self.VIDEO_PATH, cv_config="configs/liteflownet.json"))
    
    # def test_webcam(self):
    #     fs = transflow.flow.FlowSource.from_args("0", size=(1280, 720))
    #     with fs:
    #         self.assertEqual(fs.width, 1280)
    #         self.assertEqual(fs.height, 720)
    #         self.assertIsNone(fs.length)
    #         flow = next(fs)
    #         self.assertEqual(flow.shape, (720, 1280, 2))

    def test_kernels(self):
        if os.path.isfile("kernels/gradxy.npy"):
            self._test_fs(transflow.flow.FlowSource.from_args(self.VIDEO_PATH, kernel_path="kernels/gradxy.npy"))
    
    def test_filters(self):
        fs = transflow.flow.FlowSource.from_args(self.VIDEO_PATH, flow_filters="scale=2*t;threshold=2*t;clip=2*t;polar=r:a")
        with fs:
            self.assertEqual(len(fs.flow_filters), 4)
        self._test_fs(fs)
