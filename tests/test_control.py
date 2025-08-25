import multiprocessing
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transflow.pipeline import Pipeline
from transflow.config import Config, PixmapSourceConfig
import extra.control


class TestEnvironment:

    def __init__(self):
        self.folder = pathlib.Path(tempfile.gettempdir()) / "transflow-tests"

    def __enter__(self):
        if self.folder.is_dir():
            shutil.rmtree(self.folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self.folder)


class TestControl(unittest.TestCase):

    def test_control(self):
        with TestEnvironment() as env:
            Pipeline(Config(
                "assets/River.mp4",
                duration_time="00:00:00.050",
                pixmap_sources=[PixmapSourceConfig("bwnoise")],
                output_path=(env.folder / "river.mp4").as_posix()),
                checkpoint_end=True, status_queue=multiprocessing.Queue()).run()
            ckpt_path = env.folder / "river_00002.ckpt.zip"
            self.assertTrue(ckpt_path.is_file())
            window = extra.control.Window(1600, ckpt_path.as_posix(), max_sources_display=10, silent=True)
            with window:
                window.draw()
                window.export(force_export_all=True)


if __name__ == "__main__":
    unittest.main()
