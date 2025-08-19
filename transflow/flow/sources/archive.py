import json
import zipfile

import numpy

from .source import FlowSource
from ...types import Flow


class ArchiveFlowSource(FlowSource):

    class Builder(FlowSource.Builder):

        def __init__(self, path: str, **kwargs):
            super().__init__(**kwargs)
            self.path = path
            self.archive = None

        @property
        def cls(self):
            return ArchiveFlowSource

        def build(self):
            self.archive = zipfile.ZipFile(self.path)
            with self.archive.open("meta.json") as file:
                data = json.loads(file.read().decode())
            # for backward compatibility, previous flows were only forward
            self.direction = FlowSource.Direction(data.get("direction", FlowSource.Direction.FORWARD.value))
            self.width = data["width"]
            self.height = data["height"]
            self.framerate = data["framerate"]
            self.base_length = len(self.archive.infolist()) - 1

        def args(self):
            return [self.archive, *FlowSource.Builder.args(self)]

    def __init__(self, archive: zipfile.ZipFile, *args, **kwargs):
        self.archive = archive
        FlowSource.__init__(self, *args, **kwargs)
    
    def validate(self):
        super().validate()
        self.assert_type("archive", zipfile.ZipFile)

    def next(self) -> Flow:
        with self.archive.open(f"{self.input_frame_index:09d}.npy") as file:
            flow = numpy.load(file)
        return flow

    def close(self):
        self.archive.close()
