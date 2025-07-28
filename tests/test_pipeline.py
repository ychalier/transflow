import multiprocessing
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transflow.pipeline import Pipeline, Config


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
        Pipeline(Config(
            flow_path,
            bitmap_path,
            output_path,
            None,
            **kwargs,
        ), status_queue=status_queue).run()
        while not status_queue.empty():
            status: Pipeline.Status = status_queue.get()
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
        return # TODO: add length / duration tests and fix this
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
        Pipeline(
            Config(
                "assets/River.mp4",
                "assets/Deer.jpg",
                (output_dir / "1-%d.png").as_posix(),
                None,
                duration_time=(ckpt + 1) / framerate,
            ),
            checkpoint_every=ckpt,
            status_queue=status_queue,
            log_handler="file",
            safe=False,
            ).run()
        while not status_queue.empty():
            status: Pipeline.Status = status_queue.get()
            self.assertIsNone(status.error)
        self.assertEqual(len(list(output_dir.glob("1-*.png"))), ckpt + 1)
        for i in range(ckpt + 1):
            self.assertTrue((output_dir / f"1-{i}.png").is_file())
            os.rename(output_dir / f"1-{i}.png", output_dir / f"2-{i}.png")
        ckpt_path = output_dir / f"1-%d_{ckpt:05d}.ckpt.zip"
        self.assertTrue(ckpt_path.is_file())
        Pipeline(Config(ckpt_path.as_posix()), status_queue=status_queue, safe=False, log_handler="file").run()
        first_status = True
        while not status_queue.empty():
            status: Pipeline.Status = status_queue.get()
            if first_status:
                self.assertEqual(status.cursor, ckpt + 1)
            first_status = False
            self.assertIsNone(status.error)
        self.assertEqual(len(list(output_dir.glob("1-*.png"))), 1)
        self.assertTrue((output_dir / f"1-{ckpt}.png").is_file())
        status_queue.close()
        import numpy
        import PIL.Image
        img_one = numpy.array(PIL.Image.open(output_dir / f"1-{ckpt}.png"))
        img_two = numpy.array(PIL.Image.open(output_dir / f"2-{ckpt}.png"))
        diff = float(numpy.average(numpy.abs(img_one - img_two)))
        self.assertEqual(diff, 0)
        shutil.rmtree(output_dir)

    def test_config_io(self):
        config = Config("assets/River.mp4", "assets/Deer.jpg", "out/Test.mp4")
        output_path = os.path.join(tempfile.gettempdir(), "test-config.json")
        with open(output_path, "w") as file:
            json.dump(config.todict(), file)
        with open(output_path, "r") as file:
            doppelganger = Config.fromdict(json.load(file))
        os.remove(output_path)
        attrs = ["flow_path", "bitmap_path", "output_path", "extra_flow_paths",
            "flows_merging_function", "use_mvs", "mask_path", "kernel_path",
            "cv_config", "flow_filters", "direction", "seek_time",
            "duration_time", "repeat", "lock_expr", "lock_mode",
            "bitmap_seek_time", "bitmap_alteration_path", "bitmap_repeat",
            "reset_mode", "reset_alpha", "reset_mask_path", "heatmap_mode",
            "heatmap_args", "heatmap_reset_threshold", "acc_method",
            "accumulator_background", "stack_composer", "initial_canvas",
            "bitmap_mask_path", "crumble", "bitmap_introduction_flags",
            "vcodec", "size", "output_intensity", "output_heatmap",
            "output_accumulator", "render_scale", "render_colors",
            "render_binary", "seed"]
        for attr in attrs:
            self.assertEqual(getattr(config, attr), getattr(doppelganger, attr))


if __name__ == "__main__":
    unittest.main()
