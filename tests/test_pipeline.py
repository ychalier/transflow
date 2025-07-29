import multiprocessing
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy
import PIL.Image

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


class TestEnvironment:

    def __init__(self):
        self.folder = pathlib.Path(tempfile.gettempdir()) / "transflow-tests"

    def __enter__(self):
        if self.folder.is_dir():
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        return self

    def run(self, config: Config, **kwargs):
        status_queue = multiprocessing.Queue()
        pipeline = Pipeline(config, status_queue=status_queue, **kwargs)
        pipeline.run()
        statuses: list[Pipeline.Status] = []
        while not status_queue.empty():
            statuses.append(status_queue.get())
        return pipeline, statuses

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self.folder)


def compute_image_diff(left: pathlib.Path, right: pathlib.Path) -> float:
    import numpy
    import PIL.Image
    img_one = numpy.array(PIL.Image.open(left))
    img_two = numpy.array(PIL.Image.open(right))
    return float(numpy.average(numpy.abs(img_one - img_two)))


class TestPipeline(unittest.TestCase):

    def _test_config(self, flow_path, bitmap_path, **kwargs):
        with TestEnvironment() as env:
            output_path = env.folder / "test.mp4"
            for status in env.run(Config(flow_path, bitmap_path=bitmap_path, output_path=output_path.as_posix(), **kwargs))[1]:
                self.assertIsNone(status.error)
            self.assertTrue(output_path.is_file())

    def test_basic(self):
        self._test_config(
            "assets/River.mp4",
            bitmap_path="assets/Deer.jpg",
            direction="backward",
            duration_time=.2)

    def test_advanced(self):
        self._test_config(
            "assets/River.mp4",
            bitmap_path="assets/Frame.png",
            direction="forward",
            reset_mode="random",
            reset_mask_path="assets/Mask.png",
            heatmap_args="0:0:0:0",
            duration_time=.2)

    def test_checkpoint(self):
        ckpt = 5
        framerate = 50
        with TestEnvironment() as env:
            pipeline, _ = env.run(
                Config("assets/River.mp4",
                    bitmap_path="assets/Deer.jpg",
                    output_path=(env.folder / "1-%d.png").as_posix(),
                    duration_time=(ckpt + 1)/framerate),
                safe=False,
                checkpoint_every=ckpt)
            self.assertEqual(pipeline.cursor, ckpt + 1)
            self.assertEqual(pipeline.expected_length, ckpt + 1)
            self.assertEqual(len(list(env.folder.glob("1-*.png"))), ckpt + 1)
            ckpt_path = env.folder / f"1-%d_{ckpt:05d}.ckpt.zip"
            self.assertTrue(ckpt_path.is_file())
            for i in range(ckpt + 1):
                self.assertTrue((env.folder / f"1-{i}.png").is_file())
                os.rename(env.folder / f"1-{i}.png", env.folder / f"2-{i}.png")
            pipeline, _ = env.run(
                Config(ckpt_path.as_posix()),
                safe=False,
                checkpoint_every=ckpt)
            self.assertEqual(pipeline.cursor, ckpt + 1)
            self.assertEqual(pipeline.expected_length, 1)
            self.assertEqual(len(list(env.folder.glob("1-*.png"))), 1)
            self.assertEqual(compute_image_diff(env.folder / f"1-{ckpt}.png", env.folder / f"2-{ckpt}.png"), 0)

    def test_config_io(self):
        config = Config("assets/River.mp4", bitmap_path="assets/Deer.jpg", output_path="out/Test.mp4")
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

    def test_cursor(self):
        with TestEnvironment() as env:
            n = 5
            framerate = 50
            pipeline, _ = env.run(Config("assets/River.mp4", bitmap_path="assets/River.mp4", output_path=(env.folder / "%d.png").as_posix(), duration_time=n/framerate))
            self.assertEqual(pipeline.cursor, n)
            self.assertEqual(pipeline.expected_length, n)

    def test_cursor_backward(self):
        with TestEnvironment() as env:
            n = 5
            framerate = 50
            pipeline, _ = env.run(Config("assets/River.mp4", bitmap_path="assets/River.mp4", output_path=(env.folder / "%d.png").as_posix(), seek_time=get_video_duration(pathlib.Path("assets/River.mp4"))-n/framerate))
            self.assertEqual(pipeline.cursor, n - 1)
            self.assertEqual(pipeline.expected_length, n - 1)


class TestTimings(unittest.TestCase):

    def test_duration(self):
        with TestEnvironment() as env:
            env.run(Config("assets/River.mp4", bitmap_path="assets/Deer.jpg", output_path=(env.folder / "test.mp4").as_posix(), duration_time=0.1))
            self.assertTrue(get_video_duration(env.folder / "test.mp4"), 0.1)

    def test_seek(self):
        with TestEnvironment() as env:
            env.run(Config("assets/River.mp4", bitmap_path="assets/River.mp4", output_path=(env.folder / "1-%d.png").as_posix(), duration_time=0.1))
            self.assertEqual(len(list(env.folder.glob("1-*.png"))), 5)
            env.run(Config("assets/River.mp4", bitmap_path="assets/River.mp4", output_path=(env.folder / "2-%d.png").as_posix(), seek_time=2, duration_time=0.1))
            self.assertEqual(len(list(env.folder.glob("2-*.png"))), 5)
            img_one = numpy.array(PIL.Image.open(env.folder / "1-0.png"))
            img_two = numpy.array(PIL.Image.open(env.folder / "2-0.png"))
            diff = float(numpy.average(numpy.abs(img_one - img_two)))
            self.assertNotEqual(diff, 0)

    def test_seek_duration(self):
        framerate = 50
        for n in [1, 5, 10]:
            with TestEnvironment() as env:
                env.run(Config("assets/River.mp4", bitmap_path="assets/River.mp4", output_path=(env.folder / "3-%d.png").as_posix(), seek_time=get_video_duration(pathlib.Path("assets/River.mp4"))-n/framerate))
                self.assertEqual(len(list(env.folder.glob("3-*.png"))), n - 1)


if __name__ == "__main__":
    unittest.main()
