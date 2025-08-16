import enum
import logging
import os
import warnings
from typing import Callable

import numpy

from ..filters import FlowFilter
from ...utils import load_mask, parse_lambda_expression


logger = logging.getLogger(__name__)


class FlowSource:

    @enum.unique
    class Direction(enum.Enum):
        FORWARD = 0 # past to present
        BACKWARD = 1 # present to past

        @classmethod
        def from_arg(cls, arg: "str | int | FlowSource.Direction | None"):
            if arg is None:
                logger.debug("Setting flow direction to default (FORWARD)")
                return FlowSource.Direction.FORWARD
            if isinstance(arg, FlowSource.Direction):
                return arg
            if isinstance(arg, int):
                return FlowSource.Direction(arg)
            if arg == "forward":
                return FlowSource.Direction.FORWARD
            if arg == "backward":
                return FlowSource.Direction.BACKWARD
            raise ValueError(f"Invalid Flow Direction: {arg}")

    @enum.unique
    class LockMode(enum.Enum):
        STAY = 0
        SKIP = 1

        @classmethod
        def from_arg(cls, arg: "str | int | FlowSource.LockMode | None"):
            if arg is None:
                return FlowSource.LockMode.STAY
            if isinstance(arg, FlowSource.LockMode):
                return arg
            if isinstance(arg, int):
                return FlowSource.LockMode(arg)
            if arg == "stay":
                return FlowSource.LockMode.STAY
            if arg == "skip":
                return FlowSource.LockMode.SKIP
            raise ValueError(f"Invalid Lock Mode: {arg}")

    class Builder:

        def __init__(self,
                direction: "str | FlowSource.Direction | None" = "backward",
                mask_path: str | None = None,
                kernel_path: str | None = None,
                flow_filters: str | None = None,
                seek_ckpt: int | None = None,
                seek_time: float | None = None,
                duration_time: float | None = None,
                repeat: int = 1,
                lock_expr: str | None = None,
                lock_mode: "str | FlowSource.LockMode | None" = "stay"):
            self.direction: FlowSource.Direction = FlowSource.Direction.from_arg(direction)
            self.width: int | None = None
            self.height: int | None = None
            self.framerate: float = 30
            self.mask_path: str | None = mask_path
            self.mask: numpy.ndarray | None = None
            self.kernel_path = kernel_path
            self.kernel: numpy.ndarray | None = None
            self.flow_filters: list[FlowFilter] = []
            self.flow_filters_string: str | None = flow_filters
            self.seek_ckpt: int | None = seek_ckpt
            self.seek_time: float | None = seek_time
            self.duration_time: float | None = duration_time
            self.is_stream: bool = False
            self.base_length: int | None = None
            self.length: int | None = None
            self.start_frame: int = 0
            self.ckpt_start_frame: int = 0
            self.end_frame: int = 0
            self.repeat: int = repeat
            self.prev_flow: numpy.ndarray | None = None
            self.lock_expr_string: str | None = lock_expr
            self.lock_expr_stay: tuple[tuple[float, float]] | None = None
            self.lock_expr_stay_index: int = 0
            self.lock_expr_skip: Callable[[float], bool] | None = None
            self.lock_mode: FlowSource.LockMode = FlowSource.LockMode.from_arg(lock_mode)
            self.lock_start: float | None = None
            self.source: FlowSource | None = None

        @property
        def cls(self) -> type["FlowSource"]:
            return FlowSource

        def args(self) -> list:
            return [
                self.direction,
                self.width,
                self.height,
                self.framerate,
                self.length,
                self.start_frame,
                self.ckpt_start_frame,
                self.end_frame
            ]

        def kwargs(self) -> dict:
            return {
                "mask": self.mask,
                "kernel": self.kernel,
                "flow_filters": self.flow_filters,
                "lock_mode": self.lock_mode,
                "lock_expr_stay": self.lock_expr_stay,
                "lock_expr_skip": self.lock_expr_skip,
            }

        def build(self):

            if self.mask_path is not None:
                self.mask = load_mask(self.mask_path, newaxis=True)

            if self.kernel_path is not None:
                self.kernel = numpy.load(self.kernel_path)

            if self.lock_expr_string is not None:
                if self.lock_mode == FlowSource.LockMode.STAY:
                    if "(" not in self.lock_expr_string:
                        self.lock_expr_string = f"({self.lock_expr_string})"
                    self.lock_expr_stay = tuple(eval(f"[{self.lock_expr_string},]"))
                elif self.lock_mode == FlowSource.LockMode.SKIP:
                    self.lock_expr_skip = parse_lambda_expression(self.lock_expr_string)

            if self.flow_filters_string is not None:
                self.flow_filters: list[FlowFilter] = []
                for filter_string in self.flow_filters_string.strip().split(";"):
                    if filter_string.split() == "":
                        continue
                    i = filter_string.index("=")
                    self.flow_filters.append(FlowFilter.from_args(
                        filter_string[:i].strip(),
                        tuple(filter_string[i+1:].strip().split(":"))))

            if self.base_length is not None and self.base_length <= 0:
                self.base_length = None
            logger.debug("Flow source Base length: %s", self.base_length)

            self.is_stream = self.base_length == None
            if self.is_stream and self.repeat > 1:
                warnings.warn("Flow source is a stream, cannot repeat it!")
                self.repeat = 1
            if self.is_stream and self.seek_time is not None and self.seek_time > 0:
                warnings.warn("Flow source is a stream, seek time is ignored!")
                self.seek_time = None
            logger.debug("Flow source Is stream: %s", self.is_stream)

            if self.seek_time is not None and not self.is_stream:
                self.start_frame = int(self.seek_time * self.framerate)
            else:
                self.start_frame = 0
            logger.debug("Flow source Start frame: %d", self.start_frame)

            if self.duration_time is not None:
                self.end_frame = self.start_frame + int(round(self.duration_time * self.framerate, 3)) # rounding before flooring to avoid float inaccuracies
                if self.base_length is not None:
                    self.end_frame = min(self.end_frame, self.base_length)
            elif self.base_length is not None:
                self.end_frame = self.base_length
            logger.debug("Flow source End frame: %s", self.end_frame)

            if self.repeat == 0:
                self.length = None
            elif self.is_stream:
                self.length = self.end_frame
            else:
                self.length = self.repeat * (self.end_frame - self.start_frame)
            logger.debug("Flow source Length: %s", self.length)

            if self.length is not None and self.lock_mode == FlowSource.LockMode.STAY and self.lock_expr_stay is not None:
                for _, lock_duration in self.lock_expr_stay:
                    self.length += int(lock_duration * self.framerate)

            real_start_frame = self.start_frame
            ckpt_start_frame = real_start_frame
            if self.seek_ckpt is not None:
                self.output_frame_index = self.seek_ckpt
                ckpt_start_frame += self.seek_ckpt % (self.end_frame - self.start_frame)
            self.ckpt_start_frame = ckpt_start_frame
            self.start_frame = real_start_frame

        def __enter__(self):
            self.build()
            self.source = self.cls(*self.args(), **self.kwargs())
            self.source.validate()
            logger.debug("Built '%s'", self.source.__class__.__name__)
            return self.source

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if self.source is not None:
                logger.debug("Closing flow source '%s'", self.source.__class__.__name__)
                self.source.close()

    def __init__(self,
            direction: Direction,
            width: int,
            height: int,
            framerate: float,
            length: int | None,
            start_frame: int,
            ckpt_start_frame: int,
            end_frame: int,
            mask: numpy.ndarray | None = None,
            kernel: numpy.ndarray | None = None,
            flow_filters: list[FlowFilter] = [],
            lock_mode: LockMode = LockMode.STAY,
            lock_expr_stay: tuple[tuple[float, float]] | None = None,
            lock_expr_skip: Callable[[float], bool] | None = None,
            ):
        self.direction = direction
        self.width = width
        self.height = height
        self.framerate = framerate
        self.length = length
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.mask = mask
        self.kernel = kernel
        self.flow_filters = flow_filters
        self.lock_mode = lock_mode
        self.lock_expr_stay = lock_expr_stay
        self.lock_expr_skip = lock_expr_skip
        self.input_frame_index: int = 0
        self.output_frame_index: int = 0
        self.prev_flow: numpy.ndarray | None = None
        self.lock_start: float | None = None
        self.lock_expr_stay_index: int = 0

        self.start_frame = ckpt_start_frame
        self.rewind()
        self.start_frame = start_frame
        
        shape = (self.height, self.width)
        self.base_flat = numpy.arange(self.height * self.width)
        self.fx_min = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fx_max = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fy_min = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fy_max = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        for i in range(self.height):
            for j in range(self.width):
                self.fx_min[i, j] = -j
                self.fx_max[i, j] = self.width - j - 1
                self.fy_min[i, j] = -i
                self.fy_max[i, j] = self.height - i - 1
        self.basex = numpy.broadcast_to(numpy.arange(self.width), shape).copy()
        self.basey = numpy.broadcast_to(numpy.arange(self.height)[:,numpy.newaxis], shape).copy()

    def __len__(self):
        return self.length

    def assert_type(self, attr: str, *types: type):
        if not any(isinstance(getattr(self, attr), t) for t in types):
            raise ValueError(f"Attribute {attr} has incorrect type {type(attr)}")

    def validate(self):
        self.assert_type("direction", FlowSource.Direction)
        self.assert_type("width", int)
        self.assert_type("height", int)
        self.assert_type("framerate", float)
        self.assert_type("length", int, type(None))
        self.assert_type("start_frame", int)
        self.assert_type("end_frame", int)
        self.assert_type("mask", numpy.ndarray, type(None))
        self.assert_type("kernel", numpy.ndarray, type(None))
        self.assert_type("flow_filters", list)
        self.assert_type("lock_mode", FlowSource.LockMode)
        self.assert_type("lock_expr_stay", tuple, type(None))

    def read_next_flow(self) -> numpy.ndarray:
        if self.input_frame_index == self.end_frame:
            self.rewind()
        flow = self.next()
        self.input_frame_index += 1
        return flow

    def __next__(self) -> numpy.ndarray:
        if self.length is not None and self.output_frame_index >= self.length:
            raise StopIteration
        locked = False
        if self.lock_mode == FlowSource.LockMode.STAY and self.lock_expr_stay is not None:
            was_locked = self.lock_start is not None
            if was_locked:
                lock_elapsed = self.t - self.lock_start # type: ignore
                locked = lock_elapsed < self.lock_expr_stay[self.lock_expr_stay_index][1]
                if not locked:
                    self.lock_expr_stay_index += 1
                    self.lock_start = None
            if (not was_locked) or (not locked):
                locked = self.t >= self.lock_expr_stay[self.lock_expr_stay_index][0]
                if locked:
                    self.lock_start = self.t
        elif self.lock_mode == FlowSource.LockMode.SKIP and self.lock_expr_skip is not None:
            locked = self.lock_expr_skip(self.t)
        if locked:
            if self.prev_flow is None:
                raise RuntimeError("Flow is locked but has not been initialized. Maybe lock the flow later?")
            flow = self.prev_flow
        else:
            flow = self.read_next_flow()
        self.prev_flow = flow
        if locked and self.lock_mode == FlowSource.LockMode.SKIP:
            self.read_next_flow()
        self.output_frame_index += 1
        return self.post_process(flow)

    @property
    def t(self) -> float:
        return 0 if self.framerate is None else self.output_frame_index / self.framerate

    def next(self) -> numpy.ndarray:
        raise NotImplementedError()

    def rewind(self):
        logger.debug("Rewinding flow source to frame %d (currently %d)", self.start_frame, self.input_frame_index)
        self.input_frame_index = self.start_frame

    def __iter__(self):
        return self

    def post_process(self, flow: numpy.ndarray) -> numpy.ndarray:
        if self.flow_filters:
            for flow_filter in self.flow_filters:
                flow_filter.apply(flow, self.t)
        if self.mask is not None:
            flow = numpy.multiply(self.mask, flow)
        if self.kernel is not None:
            import scipy.signal
            flow_filtered_x = scipy.signal.convolve2d(flow[:,:,0], self.kernel, mode="same", boundary="fill", fillvalue=0)
            flow_filtered_y = scipy.signal.convolve2d(flow[:,:,1], self.kernel, mode="same", boundary="fill", fillvalue=0)
            flow = numpy.stack([flow_filtered_x, flow_filtered_y], axis=-1)
        if self.direction == FlowSource.Direction.FORWARD:
            numpy.clip(flow[:,:,0], self.fx_min, self.fx_max, flow[:,:,0])
            numpy.clip(flow[:,:,1], self.fy_min, self.fy_max, flow[:,:,1])
            flow_int = numpy.round(flow).astype(numpy.int32)
            flow_flat = numpy.ravel(flow_int[:,:,1] * self.width + flow_int[:,:,0])
            where = numpy.nonzero(flow_flat)
            Ax = self.basex.copy()
            Ay = self.basey.copy()
            numpy.put(Ax, self.base_flat[where] + flow_flat[where], Ax.flat[where], mode="clip")
            numpy.put(Ay, self.base_flat[where] + flow_flat[where], Ay.flat[where], mode="clip")
            flow[:,:,0] = Ax - self.basex
            flow[:,:,1] = Ay - self.basey
        numpy.clip(flow[:,:,0], self.fx_min, self.fx_max, flow[:,:,0])
        numpy.clip(flow[:,:,1], self.fy_min, self.fy_max, flow[:,:,1])
        return flow

    @classmethod
    def from_args(cls,
            flow_path: str,
            use_mvs: bool = False,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            cv_config: str | None = None,
            flow_filters: str | None = None,
            size: tuple[int, int] | None = None,
            direction: Direction | None = None,
            seek_ckpt: int | None = None,
            seek_time: float | None = None,
            duration_time: float | None = None,
            repeat: int = 1,
            lock_expr: str | None = None,
            lock_mode: str | LockMode = LockMode.STAY):
        if "::" in flow_path:
            avformat, file = flow_path.split("::")
        else:
            avformat, file = None, flow_path
        kwargs = {
            "direction": direction,
            "mask_path": mask_path,
            "kernel_path": kernel_path,
            "flow_filters": flow_filters,
            "seek_ckpt": seek_ckpt,
            "seek_time": seek_time,
            "duration_time": duration_time,
            "repeat": repeat,
            "lock_expr": lock_expr,
            "lock_mode": lock_mode
        }
        if file.endswith(".flow.zip"):
            from .archive import ArchiveFlowSource
            return ArchiveFlowSource.Builder(file, **kwargs)
        elif use_mvs:
            from .av import AvFlowSource
            return AvFlowSource.Builder(file, avformat, **kwargs)
        else:
            from .cv import CvFlowConfig, CvFlowSource
            if cv_config == "window":
                config = CvFlowConfig(show_window=True)
            elif cv_config is not None and os.path.isfile(cv_config):
                config = CvFlowConfig.from_file(cv_config)
            else:
                config = CvFlowConfig()
            return CvFlowSource.Builder(file, config, size, **kwargs)

    def close(self):
        pass

