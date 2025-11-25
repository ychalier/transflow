from typing import cast

import numpy

from .source import FlowSource
from ...types import Flow


class DummyFlowSource(FlowSource):

    class Builder(FlowSource.Builder):

        def __init__(self, width: int, height: int, value: float, framerate: float = 10, length: int = 10, **kwargs):
            super().__init__(**kwargs)
            self.target_width = width
            self.target_height = height
            self.target_value = value
            self.target_framerate = float(framerate)
            self.target_length = length
            self.array: Flow | None = None

        @property
        def cls(self):
            return DummyFlowSource

        def build(self):
            self.direction = FlowSource.Direction.BACKWARD
            self.width = self.target_width
            self.height = self.target_height
            self.framerate = self.target_framerate
            self.base_length = self.target_length
            self.array = cast(Flow, numpy.ones((self.height, self.width, 2), dtype=numpy.float32) * self.target_value)
            super().build()

        def args(self):
            return [self.array, *FlowSource.Builder.args(self)]

    def __init__(self, array: Flow, *args, **kwargs):
        self.array = array
        FlowSource.__init__(self, *args, **kwargs)

    def next(self) -> Flow:
        return cast(Flow, numpy.copy(self.array))
