import multiprocessing
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import transflow.pipeline


def get_video_duration(path: pathlib.Path) -> float:
    data = json.loads(subprocess.check_output([
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        path.as_posix()]))
    return float(data["format"]["duration"])


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
        ), status_queue=status_queue, log_level="CRITICAL")
        while not status_queue.empty():
            status: transflow.pipeline.Status = status_queue.get()
            self.assertIsNone(status.error)
        status_queue.close()
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

    def test_checkpoint(self):
        # TODO: make the test environment easier to use
        import shutil
        from pathlib import Path
        output_dir = Path(tempfile.gettempdir()) / "test-ckpt"
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        status_queue = multiprocessing.Queue()
        ckpt = 5
        framerate = 50
        duration_one = (ckpt + 1) / framerate
        duration_two = .2
        transflow.pipeline.transfer(transflow.pipeline.Config(
            "assets/River.mp4",
            "assets/Deer.jpg",
            (output_dir / "1.mp4").as_posix(),
            None,
            duration_time=duration_one,
            checkpoint_every=ckpt,
        ), status_queue=status_queue, log_level="CRITICAL")
        while not status_queue.empty():
            status: transflow.pipeline.Status = status_queue.get()
            self.assertIsNone(status.error)
        ckpt_path = output_dir / f"1_{ckpt:05d}.ckpt.zip"
        self.assertTrue(ckpt_path.is_file())
        transflow.pipeline.transfer(transflow.pipeline.Config(
            ckpt_path.as_posix(),
            "assets/Deer.jpg",
            (output_dir / "2.mp4").as_posix(),
            None,
            duration_time=duration_two,
        ), status_queue=status_queue, log_level="CRITICAL")
        first_status = True
        while not status_queue.empty():
            status: transflow.pipeline.Status = status_queue.get()
            if first_status:
                self.assertEqual(status.cursor, ckpt + 1)
            first_status = False
            self.assertIsNone(status.error)
        self.assertTrue((output_dir / "2.mp4").is_file())
        status_queue.close()
        # TODO: check the amount of frames produced
        # TODO: check that the last frame of 1 is the same as the first of 2
        shutil.rmtree(output_dir)


# TODO: add main function to run file manually