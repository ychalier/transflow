import os
import pathlib
import shutil
import tempfile
import unittest

import numpy

from transflow.output import VideoOutput


class TestOutput(unittest.TestCase):

    WIDTH = 854
    HEIGHT = 480
    FPS = 50

    def _test_output(self, output: VideoOutput):
        self.assertEqual(output.width, self.WIDTH)
        self.assertEqual(output.height, self.HEIGHT)
        frame = numpy.zeros((self.HEIGHT, self.WIDTH, 3), dtype=numpy.uint8)
        output.feed(frame)
        
    def test_ffmpeg_output(self):
        path = os.path.join(tempfile.gettempdir(), "test.mp4")
        if os.path.isfile(path):
            os.remove(path)
        self.assertFalse(os.path.isfile(path))
        output = VideoOutput.from_args(path, self.WIDTH, self.HEIGHT, self.FPS, replace=True, safe=False)
        import transflow.output.ffmpeg
        self.assertIsInstance(output, transflow.output.ffmpeg.FFmpegVideoOutput)
        with output:
            self._test_output(output)
        self.assertTrue(os.path.isfile(path))
        os.remove(path)
    
    def test_cv_output(self):
        output = VideoOutput.from_args(None, self.WIDTH, self.HEIGHT, replace=True, safe=False)
        import transflow.output.cv
        self.assertIsInstance(output, transflow.output.cv.CvVideoOutput)
        with output:
            self._test_output(output)
    
    def test_mjpeg_output(self):
        host = "localhost"
        port = 8001
        output = VideoOutput.from_args(f"mjpeg:{port}:{host}", self.WIDTH, self.HEIGHT, self.FPS)
        import transflow.output.mjpeg
        self.assertIsInstance(output, transflow.output.mjpeg.MjpegOutput)
        from aiohttp.web_runner import GracefulExit
        import socket
        try:
            with output:
                self._test_output(output)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((host, port))
                self.assertEqual(result, 0)
                sock.close()
        except GracefulExit:
            pass

    def test_frames_output(self):
        directory = pathlib.Path(tempfile.gettempdir()) / "test-frames"
        if directory.is_dir():
            shutil.rmtree(directory)
        self.assertFalse(directory.is_dir())
        path = os.path.join(directory, "%03d.png")
        output = VideoOutput.from_args(path, self.WIDTH, self.HEIGHT, self.FPS, replace=True, safe=False)
        import transflow.output.frames
        self.assertIsInstance(output, transflow.output.frames.FramesVideoOutput)
        with output:
            self._test_output(output)
        self.assertTrue(directory.is_dir())
        self.assertEqual(len(list(directory.glob("*.png"))), 1)
        shutil.rmtree(directory)
