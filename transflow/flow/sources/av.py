from typing import Optional, cast

import av.container
import numpy

from .source import FlowSource
from ...types import Flow


class AvFlowSource(FlowSource):

    class Builder(FlowSource.Builder):

        def __init__(self,
                file: str,
                avformat: str | None = None,
                **kwargs):
            super().__init__(**kwargs)
            self.file = file
            self.avformat = avformat
            self.container = None
            self.iterator = None

        @property
        def cls(self):
            return AvFlowSource

        def build(self):
            self.container = av.container.open(format=self.avformat, file=self.file)
            context = self.container.streams.video[0].codec_context
            context.options = {"flags2": "+export_mvs"}
            first_frame = next(self.container.decode(video=0))
            self.width = first_frame.width
            self.height = first_frame.height
            if context.framerate:
                self.framerate = float(context.framerate)
            self.base_length = self.container.streams.video[0].frames - 1
            super().build()

        def args(self):
            return [self.container, *FlowSource.Builder.args(self)]

    def __init__(self,
            container: av.container.InputContainer,
            *args,
            **kwargs):
        self.container = container
        self.iterator = self.container.decode(video=0)
        FlowSource.__init__(self, *args, **kwargs)
    
    def validate(self):
        super().validate()
        self.assert_type("container", av.container.InputContainer)

    def rewind(self):
        FlowSource.rewind(self)
        self.container.seek(0)
        for _ in range(self.input_frame_index + 1):
            next(self.iterator)

    def next(self) -> Flow:
        flow = cast(Flow, numpy.zeros((self.height, self.width, 2), dtype=numpy.float32))
        frame = next(self.iterator)
        from av.sidedata.motionvectors import MotionVectors
        vectors = cast(Optional[MotionVectors], frame.side_data.get("MOTION_VECTORS"))
        if vectors is None:
            return flow
        for mv in vectors:
            assert mv.source == -1, "Encode with bf=0 and refs=1"
            i0 = mv.src_y - mv.h // 2
            i1 = mv.src_y + mv.h // 2
            j0 = mv.src_x - mv.w // 2
            j1 = mv.src_x + mv.w // 2
            dx = mv.motion_x / mv.motion_scale
            dy = mv.motion_y / mv.motion_scale
            flow[i0:i1, j0:j1] = -dx, -dy
        return flow

    def close(self):
        self.container.close()