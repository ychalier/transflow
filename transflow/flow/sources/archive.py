import json
import zipfile

import numpy

from .source import FlowSource


class ArchiveFlowSource(FlowSource):

    def __init__(self, path: str, **src_args):
        FlowSource.__init__(self, FlowSource.FlowDirection.FORWARD, **src_args)
        self.path = path
        self.archive = None

    def setup(self):
        FlowSource.setup(self)
        self.archive = zipfile.ZipFile(self.path)
        with self.archive.open("meta.json") as file:
            data = json.loads(file.read().decode())
        # for backward compatibility, previous flows were only forward
        self.direction = FlowSource.FlowDirection(data.get("direction", FlowSource.FlowDirection.FORWARD.value))
        self.set_metadata(
            data["width"],
            data["height"],
            data["framerate"],
            len(self.archive.infolist()) - 1
        )

    def rewind(self):
        self.input_frame_index = self.start_frame

    def next(self) -> numpy.ndarray:
        if self.archive is None:
            raise ValueError("Archive not initialized")
        with self.archive.open(f"{self.input_frame_index:09d}.npy") as file:
            flow = numpy.load(file)
        return flow

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.archive is not None:
            self.archive.close()
