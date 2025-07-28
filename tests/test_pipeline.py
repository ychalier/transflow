import multiprocessing
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import transflow.pipeline
import transflow.config


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
        transflow.pipeline.transfer(transflow.config.Config(
            flow_path,
            bitmap_path,
            output_path,
            None,
            **kwargs,
        ), status_queue=status_queue, log_level="CRITICAL")
        while not status_queue.empty():
            status: transflow.pipeline.Pipeline.Status = status_queue.get()
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
        transflow.pipeline.transfer(transflow.config.Config(
            "assets/River.mp4",
            "assets/Deer.jpg",
            (output_dir / "1-%d.png").as_posix(),
            None,
            duration_time=duration_one,
        ), checkpoint_every=ckpt, status_queue=status_queue, log_level="CRITICAL")
        while not status_queue.empty():
            status: transflow.pipeline.Pipeline.Status = status_queue.get()
            self.assertIsNone(status.error)
        self.assertEqual(len(list(output_dir.glob("1-*.png"))), ckpt + 1)
        self.assertTrue((output_dir / f"1-{ckpt}.png").is_file())
        ckpt_path = output_dir / f"1-%d_{ckpt:05d}.ckpt.zip"
        self.assertTrue(ckpt_path.is_file())
        transflow.pipeline.transfer(transflow.config.Config(
            ckpt_path.as_posix(),
            "assets/Deer.jpg",
            (output_dir / "2-%d.png").as_posix(),
            None,
            duration_time=duration_two,
        ), status_queue=status_queue, log_level="CRITICAL")
        first_status = True
        while not status_queue.empty():
            status: transflow.pipeline.Pipeline.Status = status_queue.get()
            if first_status:
                self.assertEqual(status.cursor, ckpt + 1)
            first_status = False
            self.assertIsNone(status.error)
        self.assertEqual(len(list(output_dir.glob("2-*.png"))), duration_two * framerate)
        self.assertTrue((output_dir / f"2-0.png").is_file())
        status_queue.close()
        import numpy
        import PIL.Image
        img_one = numpy.array(PIL.Image.open(output_dir / f"1-{ckpt}.png"))
        img_two = numpy.array(PIL.Image.open(output_dir / f"2-0.png"))
        diff = float(numpy.average(numpy.abs(img_one - img_two)))
        self.assertEqual(diff, 0)
        shutil.rmtree(output_dir)


if __name__ == "__main__":   
    unittest.main()