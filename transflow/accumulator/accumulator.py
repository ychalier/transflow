import enum

import numpy

from ..utils import load_mask
from ..flow import Direction


class Accumulator:

    @enum.unique
    class HeatmapMode(enum.Enum):

        DISCRETE = 0
        CONTINUOUS = 1

        @classmethod
        def from_string(cls, string):
            match string:
                case "discrete":
                    return cls.DISCRETE
                case "continuous":
                    return cls.CONTINUOUS
            raise ValueError(f"Unknown heatmap mode {string}")


    @enum.unique
    class ResetMode(enum.Enum):

        OFF = 0
        RANDOM = 1
        LINEAR = 2

        @classmethod
        def from_string(cls, string):
            match string:
                case "off":
                    return cls.OFF
                case "random":
                    return cls.RANDOM
                case "linear":
                    return cls.LINEAR
            raise ValueError(f"Unknown reset mode {string}")

    def __init__(self,
            width: int,
            height: int,
            heatmap_mode: HeatmapMode | str = "discrete",
            heatmap_args: str | tuple[int|float, ...] = "0:0:0:0",
            heatmap_reset_threshold: float | None = None,
            reset_mode: ResetMode | str = "off",
            reset_alpha: float = .9,
            reset_mask_path: str | None = None):
        self.width = width
        self.height = height
        self.flow_int: numpy.ndarray = numpy.zeros((height, width, 2))
        self.flow_flat: numpy.ndarray = numpy.zeros((height * width * 2,))
        self.heatmap_mode = Accumulator.HeatmapMode.from_string(heatmap_mode)\
            if isinstance(heatmap_mode, str) else heatmap_mode
        self.heatmap_reset_threshold = heatmap_reset_threshold if heatmap_reset_threshold is not None else float("inf")
        if self.heatmap_mode == Accumulator.HeatmapMode.DISCRETE:
            self.heatmap = numpy.zeros((self.height, self.width), dtype=numpy.int32)
            x = tuple(map(int, heatmap_args.split(":")))\
                if isinstance(heatmap_args, str) else heatmap_args
            self.heatmap_min = x[0]
            self.heatmap_max = x[1]
            self.heatmap_add = x[2]
            self.heatmap_sub = x[3]
        elif self.heatmap_mode == Accumulator.HeatmapMode.CONTINUOUS:
            self.heatmap = numpy.zeros((self.height, self.width), dtype=numpy.float32)
            x = tuple(map(float, heatmap_args.split(":")))\
                if isinstance(heatmap_args, str) else heatmap_args
            self.heatmap_min = 0
            self.heatmap_max = x[0]
            self.heatmap_decay = x[1]
            self.heatmap_threshold = x[2]
        self.fx_min = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fx_max = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fy_min = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        self.fy_max = numpy.zeros((self.height, self.width), dtype=numpy.int32)
        for i in range(self.height):
            for j in range(self.width):
                self.fx_min[i, j] = -j
                self.fx_max[i, j] = width - j - 1
                self.fy_min[i, j] = -i
                self.fy_max[i, j] = height - i - 1
        self.reset_mode = reset_mode if isinstance (reset_mode, Accumulator.ResetMode) else Accumulator.ResetMode.from_string(reset_mode)
        self.reset_alpha = reset_alpha
        self.reset_mask = None if reset_mask_path is None else load_mask(reset_mask_path)

    def _update_flow(self, flow: numpy.ndarray):
        numpy.clip(flow[:,:,0], self.fx_min, self.fx_max, flow[:,:,0])
        numpy.clip(flow[:,:,1], self.fy_min, self.fy_max, flow[:,:,1])
        self.flow_int = numpy.round(flow).astype(numpy.int32)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])
        if self.heatmap_mode == Accumulator.HeatmapMode.DISCRETE:
            self.heatmap = numpy.clip(self.heatmap - self.heatmap_sub,
                                      self.heatmap_min, self.heatmap_max)
            self.heatmap.flat[numpy.nonzero(self.flow_flat)] += self.heatmap_add
            self.heatmap = numpy.clip(self.heatmap, self.heatmap_min, self.heatmap_max)
        elif self.heatmap_mode == Accumulator.HeatmapMode.CONTINUOUS:
            self.heatmap *= self.heatmap_decay
            self.heatmap[numpy.where(self.heatmap < self.heatmap_threshold)] = self.heatmap_min
            self.heatmap = numpy.clip(self.heatmap + numpy.linalg.norm(self.flow_int, axis=2),
                                      self.heatmap_min, self.heatmap_max)

    def update(self, flow: numpy.ndarray, direction: Direction):
        raise NotImplementedError()

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    @classmethod
    def from_args(cls,
            width: int,
            height: int,
            method: str = "map",
            reset_mode: ResetMode | str = "off",
            reset_alpha: float = .9,
            reset_mask_path: str | None = None,
            heatmap_mode: HeatmapMode | str = "discrete",
            heatmap_args: str | tuple[int|float] = "0:0:0:0",
            heatmap_reset_threshold: float | None = None,
            bg_color: str = "ffffff",
            stack_composer: str = "top",
            initial_canvas: str | None = None,
            bitmap_mask_path: str | None = None,
            crumble: bool = False,
            bitmap_introduction_flags: int = 1):
        args = {
            "heatmap_mode": heatmap_mode,
            "heatmap_args": heatmap_args,
            "heatmap_reset_threshold": heatmap_reset_threshold,
            "reset_mode": reset_mode,
            "reset_alpha": reset_alpha,
            "reset_mask_path": reset_mask_path,
        }
        if isinstance(reset_mode, str):
            reset_mode = Accumulator.ResetMode.from_string(reset_mode)
        match method:
            case "map":
                from .mapping import MappingAccumulator
                return MappingAccumulator(width, height, **args)
            case "stack":
                from .stack import StackAccumulator
                return StackAccumulator(
                    width, height,
                    bg_color,
                    stack_composer,
                    **args)
            case "sum":
                from .sum import SumAccumulator
                return SumAccumulator(width, height, **args)
            case "crumble":
                from .crumble import CrumbleAccumulator
                return CrumbleAccumulator(width, height, bg_color, **args)
            case "canvas":
                from .canvas import CanvasAccumulator
                return CanvasAccumulator(width, height,
                    initial_canvas,
                    bitmap_mask_path,
                    crumble,
                    bitmap_introduction_flags,
                    **args)
        raise ValueError(f"Unknown accumulator method '{method}'")

    def get_heatmap_array(self) -> numpy.ndarray:
        if self.heatmap_min == self.heatmap_max:
            return numpy.zeros((self.height, self.width))
        return (self.heatmap - self.heatmap_min) / (self.heatmap_max - self.heatmap_min)

    def get_accumulator_array(self) -> numpy.ndarray:
        raise NotImplementedError()
