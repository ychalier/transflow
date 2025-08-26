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
from transflow.pipeline import Pipeline
from transflow.config import Config, PixmapSourceConfig, LayerConfig


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

    def test_basic(self):
        with TestEnvironment() as env:
            output_path = env.folder / "test.mp4"
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/Deer.jpg")],
                output_path=output_path.as_posix(),
                direction="backward",
                duration_time=0.2)
            for status in env.run(config)[1]:
                self.assertIsNone(status.error)
            self.assertTrue(output_path.is_file())

    def test_advanced(self):
        with TestEnvironment() as env:
            output_path = env.folder / "test.mp4"
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/Frame.png", layers=[0])],
                layers=[LayerConfig(0, "moveref", reset_mode="random", reset_random_factor=1, reset_mask="assets/Mask.png")],
                output_path=output_path.as_posix(),
                direction="forward",
                duration_time=0.2)
            for status in env.run(config)[1]:
                self.assertIsNone(status.error)
            self.assertTrue(output_path.is_file())

    def test_checkpoint(self):
        ckpt = 5
        framerate = 50
        with TestEnvironment() as env:
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/Deer.jpg")],
                output_path=(env.folder / "1-%d.png").as_posix(),
                direction="forward",
                duration_time=(ckpt + 1)/framerate)
            pipeline, _ = env.run(
                config,
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
        config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/Deer.jpg")],
                output_path="out/Test.mp4")
        output_path = os.path.join(tempfile.gettempdir(), "test-config.json")
        with open(output_path, "w") as file:
            json.dump(config.todict(), file)
        with open(output_path, "r") as file:
            doppelganger = Config.fromdict(json.load(file))
        os.remove(output_path)
        attrs = ["flow_path", "output_path", "extra_flow_paths",
            "flows_merging_function", "use_mvs", "mask_path", "kernel_path",
            "cv_config", "flow_filters", "direction", "seek_time",
            "duration_time", "repeat", "lock_expr", "lock_mode",
            "compositor_background",
            "vcodec", "size", "view_flow", "view_flow_magnitude", "render_scale", "render_colors",
            "render_binary", "seed"]
        for attr in attrs:
            self.assertEqual(getattr(config, attr), getattr(doppelganger, attr))
        self.assertEqual(len(config.pixmap_sources), len(doppelganger.pixmap_sources))
        attrs = ["path", "seek_time", "alteration_path", "introduction_path", "repeat", "layers"]
        for a, b in zip(config.pixmap_sources, doppelganger.pixmap_sources):
            for attr in attrs:
                self.assertEqual(getattr(a, attr), getattr(b, attr))
        self.assertEqual(len(config.layers), len(doppelganger.layers))
        attrs = ["index", "classname", "mask_src", "mask_dst", "mask_alpha",
            "transparent_pixels_can_move", "pixels_can_move_to_empty_spot",
            "pixels_can_move_to_filled_spot", "moving_pixels_leave_empty_spot",
            "reset_mode", "reset_mask", "reset_random_factor",
            "reset_constant_step", "reset_linear_factor", "reset_source",
            "introduce_pixels_on_empty_spots", "introduce_pixels_on_filled_spots",
            "introduce_moving_pixels", "introduce_unmoving_pixels", "introduce_once",
            "introduce_on_all_filled_spots", "introduce_on_all_empty_spots"]
        for a, b in zip(config.layers, doppelganger.layers):
            for attr in attrs:
                self.assertEqual(getattr(a, attr), getattr(b, attr))

    def test_cursor(self):
        with TestEnvironment() as env:
            n = 5
            framerate = 50
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/River.mp4")],
                output_path=(env.folder / "%d.png").as_posix(),
                duration_time=n/framerate)
            pipeline, _ = env.run(config)
            self.assertEqual(pipeline.cursor, n)
            self.assertEqual(pipeline.expected_length, n)

    def test_cursor_backward(self):
        with TestEnvironment() as env:
            n = 5
            framerate = 50
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/River.mp4")],
                output_path=(env.folder / "%d.png").as_posix(),
                seek_time=get_video_duration(pathlib.Path("assets/River.mp4"))-n/framerate,
                duration_time=n/framerate)
            pipeline, _ = env.run(config)
            self.assertEqual(pipeline.cursor, n - 1)
            self.assertEqual(pipeline.expected_length, n - 1)

    def test_view_flow(self):
        with TestEnvironment() as env:
            config = Config(
                "assets/River.mp4",
                output_path=(env.folder / "out.mp4").as_posix(),
                view_flow=True,
                duration_time=0.1)
            env.run(config)
            self.assertTrue((env.folder / "out.mp4").is_file())
    
    def test_view_flow_magnitude(self):
        with TestEnvironment() as env:
            config = Config(
                "assets/River.mp4",
                output_path=(env.folder / "out.mp4").as_posix(),
                view_flow_magnitude=True,
                duration_time=0.1)
            env.run(config)
            self.assertTrue((env.folder / "out.mp4").is_file())

class TestTimings(unittest.TestCase):

    def test_duration(self):
        with TestEnvironment() as env:
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/Deer.jpg")],
                output_path=(env.folder / "test.mp4").as_posix(),
                duration_time=0.1)
            env.run(config)
            self.assertEqual(get_video_duration(env.folder / "test.mp4"), 0.1)

    def test_seek(self):
        with TestEnvironment() as env:
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/River.mp4")],
                output_path=(env.folder / "1-%d.png").as_posix(),
                duration_time=0.1)
            env.run(config)
            self.assertEqual(len(list(env.folder.glob("1-*.png"))), 5)
            config = Config(
                "assets/River.mp4",
                pixmap_sources=[PixmapSourceConfig("assets/River.mp4")],
                output_path=(env.folder / "2-%d.png").as_posix(),
                seek_time=2,
                duration_time=0.1)
            env.run(config)
            self.assertEqual(len(list(env.folder.glob("2-*.png"))), 5)
            img_one = numpy.array(PIL.Image.open(env.folder / "1-0.png"))
            img_two = numpy.array(PIL.Image.open(env.folder / "2-0.png"))
            diff = float(numpy.average(numpy.abs(img_one - img_two)))
            self.assertNotEqual(diff, 0)

    def test_seek_duration(self):
        framerate = 50
        for n in [1, 5, 10]:
            with TestEnvironment() as env:
                config = Config(
                    "assets/River.mp4",
                    pixmap_sources=[PixmapSourceConfig("assets/River.mp4")],
                    output_path=(env.folder / "3-%d.png").as_posix(),
                    seek_time=get_video_duration(pathlib.Path("assets/River.mp4"))-n/framerate)
                env.run(config)
                self.assertEqual(len(list(env.folder.glob("3-*.png"))), n - 1)


if __name__ == "__main__":
    unittest.main()
