import numpy

from .source import FlowSource


class AvFlowSource(FlowSource):

    def __init__(self, file: str, avformat: str | None = None, **src_args):
        FlowSource.__init__(self, FlowSource.FlowDirection.FORWARD, **src_args)
        self.file = file
        self.avformat = avformat
        self.container = None
        self.iterator = None

    def setup(self):
        FlowSource.setup(self)
        import av.container
        self.container = av.container.open(format=self.avformat, file=self.file)
        context = self.container.streams.video[0].codec_context
        context.options = {"flags2": "+export_mvs"}
        self.iterator = self.container.decode(video=0)
        first_frame = next(self.iterator)
        self.set_metadata(
            first_frame.width,
            first_frame.height,
            float(context.framerate) if context.framerate is not None else 30,
            self.container.streams.video[0].frames - 1
        )

    def rewind(self):
        self.input_frame_index = self.start_frame
        if self.container is None:
            raise ValueError("Container not initialized")
        self.container.seek(0)
        if self.iterator is None:
            raise ValueError("Iterator not initialized")
        for _ in range(self.input_frame_index + 1):
            next(self.iterator)

    def next(self) -> numpy.ndarray:
        if self.height is None or self.width is None:
            raise ValueError("")
        flow = numpy.zeros((self.height, self.width, 2), dtype=numpy.float32)
        if self.iterator is None:
            raise ValueError("Iterator not inizalized")
        frame = next(self.iterator)
        from typing import Optional, cast
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

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.container is not None:
            self.container.close()