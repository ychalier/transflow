import multiprocessing
import os
import tempfile
import unittest

import transflow.pipeline


class TestPipeline(unittest.TestCase):

    def _test_config(self, flow_path, bitmap_path, **kwargs):
        output_path = os.path.join(tempfile.gettempdir(), "test.mp4")
        if os.path.isfile(output_path):
            os.remove(output_path)
        self.assertFalse(os.path.isfile(output_path))
        status_queue = multiprocessing.Queue()
        transflow.pipeline.transfer(transflow.pipeline.Config(
            flow_path,
            bitmap_path,
            output_path,
            None,
            **kwargs,
        ), None, status_queue)
        while not status_queue.empty():
            status: transflow.pipeline.Status = status_queue.get()
            self.assertIsNone(status.error)
        self.assertTrue(os.path.isfile(output_path))
        os.remove(output_path)

    def test_basic(self):
        self._test_config(
            "assets/River.mp4",
            "assets/Deer.jpg",
            direction="backward",
            duration_time=.2)

    def test_advanced(self):
        self._test_config(
            "assets/River.mp4",
            "assets/Frame.png",
            direction="forward",
            reset_mode="random",
            reset_mask_path="assets/Mask.png",
            heatmap_args="0:0:0:0",
            duration_time=.2)