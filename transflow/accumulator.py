import enum
import logging

import numpy

from .flow import FlowDirection
from .utils import parse_hex_color, compose_top, compose_additive,\
    compose_subtractive, compose_average, load_mask


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


class Accumulator:

    def __init__(self, width: int, height: int,
                 heatmap_mode: HeatmapMode | str = "discrete",
                 heatmap_args: str | tuple[int|float] = "0:0:0:0"):
        self.width = width
        self.height = height
        self.flow_int: numpy.ndarray = None
        self.flow_flat: numpy.ndarray = None
        self.heatmap_mode = HeatmapMode.from_string(heatmap_mode)\
            if isinstance(heatmap_mode, str) else heatmap_mode
        if self.heatmap_mode == HeatmapMode.DISCRETE:
            self.heatmap = numpy.zeros((self.height, self.width), dtype=int)
            x = tuple(map(int, heatmap_args.split(":")))\
                if isinstance(heatmap_args, str) else heatmap_args
            self.heatmap_min = x[0]
            self.heatmap_max = x[1]
            self.heatmap_add = x[2]
            self.heatmap_sub = x[3]
        elif self.heatmap_mode == HeatmapMode.CONTINUOUS:
            self.heatmap = numpy.zeros((self.height, self.width), dtype=float)
            x = tuple(map(float, heatmap_args.split(":")))\
                if isinstance(heatmap_args, str) else heatmap_args
            self.heatmap_min = 0
            self.heatmap_max = x[0]
            self.heatmap_decay = x[1]
            self.heatmap_threshold = x[2]
        self.fx_min = numpy.zeros((self.height, self.width), dtype=int)
        self.fx_max = numpy.zeros((self.height, self.width), dtype=int)
        self.fy_min = numpy.zeros((self.height, self.width), dtype=int)
        self.fy_max = numpy.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                self.fx_min[i, j] = -j
                self.fx_max[i, j] = width - j - 1
                self.fy_min[i, j] = -i
                self.fy_max[i, j] = height - i - 1

    def _update_flow(self, flow: numpy.ndarray):
        numpy.clip(flow[:,:,0], self.fx_min, self.fx_max, flow[:,:,0])
        numpy.clip(flow[:,:,1], self.fy_min, self.fy_max, flow[:,:,1])
        self.flow_int = numpy.round(flow).astype(int)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])
        if self.heatmap_mode == HeatmapMode.DISCRETE:
            self.heatmap = numpy.clip(self.heatmap - self.heatmap_sub,
                                      self.heatmap_min, self.heatmap_max)
            self.heatmap.flat[numpy.nonzero(self.flow_flat)] += self.heatmap_add
            self.heatmap = numpy.clip(self.heatmap, self.heatmap_min, self.heatmap_max)
        elif self.heatmap_mode == HeatmapMode.CONTINUOUS:
            self.heatmap *= self.heatmap_decay
            self.heatmap[numpy.where(self.heatmap < self.heatmap_threshold)] = self.heatmap_min
            self.heatmap = numpy.clip(self.heatmap + numpy.linalg.norm(self.flow_int, axis=2),
                                      self.heatmap_min, self.heatmap_max)

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        raise NotImplementedError()

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError()

    @classmethod
    def from_args(cls, width: int, height: int, method: str = "map",
                  reset_mode: ResetMode | str = "off", reset_alpha: float = .9,
                  reset_mask_path: str | None = None,
                  heatmap_mode: HeatmapMode | str = "discrete",
                  heatmap_args: str | tuple[int|float] = "0:0:0:0",
                  bg_color: str = "ffffff", stack_composer: str = "top"):
        if isinstance(reset_mode, str):
            reset_mode = ResetMode.from_string(reset_mode)
        match method:
            case "map":
                return MappingAccumulator(
                    width, height, reset_mode, reset_alpha, reset_mask_path,
                    heatmap_mode, heatmap_args)
            case "stack":
                return StackAccumulator(
                    width, height, bg_color, stack_composer,
                    heatmap_mode, heatmap_args)
            case "sum":
                return SumAccumulator(
                    width, height, reset_mode, reset_alpha, reset_mask_path,
                    heatmap_mode, heatmap_args)
            case "crumble":
                return CrumbleAccumulator(width, height, reset_mode,
                    reset_alpha, reset_mask_path, heatmap_mode, heatmap_args,
                    bg_color)
        raise ValueError(f"Unknown accumulator method '{method}'")

    def get_heatmap_array(self) -> numpy.ndarray:
        if self.heatmap_min == self.heatmap_max:
            return numpy.zeros((self.height, self.width))
        return (self.heatmap - self.heatmap_min) / (self.heatmap_max - self.heatmap_min)

    def get_accumulator_array(self) -> numpy.ndarray:
        raise NotImplementedError()


class MappingAccumulator(Accumulator):

    def __init__(self, width: int, height: int,
                 reset_mode: ResetMode = ResetMode.OFF, reset_alpha: float = .9,
                 reset_mask_path: str | None = None,
                 heatmap_mode: HeatmapMode | str = "discrete",
                 heatmap_args: str | tuple[int|float] = "0:0:0:0"):
        Accumulator.__init__(self, width, height, heatmap_mode, heatmap_args)
        self.reset_mode = reset_mode
        self.reset_alpha = reset_alpha
        shape = (self.height, self.width)
        self.reset_mask = None if reset_mask_path is None else load_mask(reset_mask_path)
        self.base_flat = numpy.arange(self.height * self.width)
        assert self.base_flat.dtype == int, self.base_flat.dtype
        self.basex = numpy.broadcast_to(numpy.arange(self.width), shape).copy()
        self.basey = numpy.broadcast_to(numpy.arange(self.height)[:,numpy.newaxis], shape).copy()
        self.mapx = self.basex.astype(float)
        self.mapy = self.basey.astype(float)

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap == 0)
                & (numpy.random.random(size=(self.height, self.width)) <= threshold))
            self.mapx[where] = self.basex[where]
            self.mapy[where] = self.basey[where]
        elif self.reset_mode == ResetMode.LINEAR:
            where = numpy.where(self.heatmap == 0)
            if self.reset_mask is None:
                self.mapx[where] = (1 - self.reset_alpha) * self.mapx[where]\
                    + self.reset_alpha * self.basex[where]
                self.mapy[where] = (1 - self.reset_alpha) * self.mapy[where]\
                    + self.reset_alpha * self.basey[where]
            else:
                self.mapx[where] = (1 - self.reset_mask[where]) * self.mapx[where]\
                    + self.reset_mask[where] * self.basex[where]
                self.mapy[where] = (1 - self.reset_mask[where]) * self.mapy[where]\
                    + self.reset_mask[where] * self.basey[where]
        if direction == FlowDirection.FORWARD:
            where = numpy.nonzero(self.flow_flat)
            numpy.put(self.mapx, self.base_flat[where] + self.flow_flat[where],
                      self.mapx.flat[where], mode="clip")
            numpy.put(self.mapy, self.base_flat[where] + self.flow_flat[where],
                      self.mapy.flat[where], mode="clip")
        elif direction == FlowDirection.BACKWARD:
            shift = (self.basey + self.flow_int[:,:,1], self.basex + self.flow_int[:,:,0])
            self.mapx = self.mapx[shift]
            self.mapy = self.mapy[shift]

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        mapping = numpy.clip(self.mapy.astype(int), 0, self.height - 1) * self.width\
            + numpy.clip(self.mapx.astype(int), 0, self.width - 1)
        out = bitmap\
            .reshape((self.height * self.width, bitmap.shape[2]))[mapping.flat]\
            .reshape(bitmap.shape)
        return out

    def get_accumulator_array(self) -> numpy.ndarray:
        return numpy.stack([self.mapx - self.basex, self.mapy - self.basey], axis=-1)


class CrumbleAccumulator(MappingAccumulator):

    def __init__(self, width: int, height: int,
                 reset_mode: ResetMode = ResetMode.OFF, reset_alpha: float = .9,
                 reset_mask_path: str | None = None,
                 heatmap_mode: HeatmapMode | str = "discrete",
                 heatmap_args: str | tuple[int|float] = "0:0:0:0",
                 bg_color: str = "000000"):
        if reset_mode not in [ResetMode.OFF, ResetMode.RANDOM]:
            logging.warning(
                "CrumbleAccumulator only works with Off or Random reset, not %s",
                reset_mode)
        MappingAccumulator.__init__(self, width, height, reset_mode,
                                    reset_alpha, reset_mask_path, heatmap_mode,
                                    heatmap_args)
        self.bg_color = parse_hex_color(bg_color)
        self.crumble_mask = numpy.ones((self.height, self.width), dtype=numpy.uint8)
    
    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap == 0)
                & (numpy.random.random(size=(self.height, self.width)) <= threshold))
            self.mapx[where] = self.basex[where]
            self.mapy[where] = self.basey[where]
            self.crumble_mask[where] = 1
        if direction == FlowDirection.FORWARD:
            where_movements = numpy.nonzero(self.flow_flat)
            w1 = numpy.nonzero(self.crumble_mask.flat)
            w3 = numpy.intersect1d(where_movements[0], w1[0])
            numpy.put(
                self.mapx,
                self.base_flat[w3] + self.flow_flat[w3],
                self.mapx.flat[w3], mode="clip")
            numpy.put(
                self.mapy,
                self.base_flat[w3] + self.flow_flat[w3],
                self.mapy.flat[w3], mode="clip")
            self.crumble_mask.flat[where_movements] = 0
            self.crumble_mask.flat[self.base_flat[w3] + self.flow_flat[w3]] = 1
        elif direction == FlowDirection.BACKWARD:
            shift = (
                self.basey + self.flow_int[:,:,1],
                self.basex + self.flow_int[:,:,0])
            w1 = numpy.nonzero(self.crumble_mask[shift])
            w1_flat = numpy.ravel(w1[0] * self.width + w1[1])
            w2 = numpy.nonzero(numpy.max(numpy.absolute(self.flow_int), axis=2))
            w2_flat = numpy.ravel(w2[0] * self.width + w2[1])
            w3 = numpy.intersect1d(w1_flat, w2_flat)
            self.mapx[w1] = self.mapx[shift][w1]
            self.mapy[w1] = self.mapy[shift][w1]
            shift_matrix = numpy.concatenate(
                [shift[0][:,:,numpy.newaxis], shift[1][:,:,numpy.newaxis]],
                axis=2)[w2]
            shift_flat = numpy.ravel(shift_matrix[:,0] * self.width + shift_matrix[:,1])
            numpy.put(self.crumble_mask, shift_flat, 0)
            self.crumble_mask.flat[w3] = 1

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        mapping = numpy.clip(self.mapy.astype(int), 0, self.height - 1) * self.width\
            + numpy.clip(self.mapx.astype(int), 0, self.width - 1)
        out = bitmap\
            .reshape((self.height * self.width, bitmap.shape[2]))[mapping.flat]\
            .reshape(bitmap.shape)
        out[self.crumble_mask == 0] = self.bg_color
        return out


class StackAccumulator(Accumulator):

    def __init__(self, width: int, height: int, bg_color: str = "ffffff",
                 composer: str = "top",
                 heatmap_mode: HeatmapMode | str = "discrete",
                 heatmap_args: str | tuple[int|float] = "0:0:0:0"):
        Accumulator.__init__(self, width, height, heatmap_mode, heatmap_args)
        self.bg_color = parse_hex_color(bg_color)
        self.composer = {
            "top": compose_top,
            "add": compose_additive,
            "sub": compose_subtractive,
            "avg": compose_average,
        }[composer]
        self.stacks = []
        for i in range(self.height):
            self.stacks.append([])
            for j in range(self.width):
                self.stacks[i].append([(i, j)])

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        for i in range(self.height):
            for j in range(self.width):
                mj = self.flow_int[i, j, 0]
                mi = self.flow_int[i, j, 1]
                if (mj == 0 and mi == 0):
                    continue
                if direction == FlowDirection.FORWARD:
                    srci, srcj = i, j
                    desti = max(0, min(self.height - 1, i + mi))
                    destj = max(0, min(self.width - 1, j + mj))
                else: # FlowDirection.BACKWARD:
                    desti, destj = i, j
                    srci = max(0, min(self.height - 1, i + mi))
                    srcj = max(0, min(self.width - 1, j + mj))
                if not self.stacks[srci][srcj]:
                    continue
                self.stacks[desti][destj].append(self.stacks[srci][srcj].pop())

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        out = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        for i in range(self.height):
            for j in range(self.width):
                if self.stacks[i][j]:
                    out[i][j] = self.composer(*[tuple(bitmap[*xy, :3]) for xy in self.stacks[i][j]])
                else:
                    out[i][j] = self.bg_color
        return out

    def get_accumulator_array(self) -> numpy.ndarray:
        arr = numpy.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                if self.stacks[i][j]:
                    mi, mj = self.stacks[i][j][-1]
                    arr[i, j] = [mj - j, mi - i]
        return arr


class SumAccumulator(Accumulator):

    def __init__(self, width: int, height: int,
                 reset_mode: ResetMode = ResetMode.OFF, reset_alpha: float = .9,
                 reset_mask_path: str | None = None,
                 heatmap_mode: HeatmapMode | str = "discrete",
                 heatmap_args: str | tuple[int|float] = "0:0:0:0"):
        Accumulator.__init__(self, width, height, heatmap_mode, heatmap_args)
        self.total_flow = numpy.zeros((height, width, 2), dtype=float)
        self.reset_mode = reset_mode
        self.reset_alpha = reset_alpha
        self.reset_mask = None if reset_mask_path is None else load_mask(reset_mask_path)
        shape = (height, width)
        self.basex = numpy.broadcast_to(numpy.arange(self.width), shape).copy()
        self.basey = numpy.broadcast_to(numpy.arange(self.height)[:,numpy.newaxis], shape).copy()

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        if direction != FlowDirection.BACKWARD:
            logging.warning("SumAccumulator only works with backward flow, not %s", direction)
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(numpy.random.random(size=(self.height, self.width)) <= threshold)
            self.total_flow[where] = 0
        elif self.reset_mode == ResetMode.LINEAR:
            if self.reset_mask is None:
                self.total_flow = (1 - self.reset_alpha) * self.total_flow
            else:
                self.total_flow = (1 - self.reset_mask) * self.total_flow
        self.total_flow += flow

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        total_flow_int = numpy.round(self.total_flow).astype(int)
        iy = self.basey + total_flow_int[:,:,1]
        numpy.clip(iy, 0, self.height - 1, iy)
        ix = self.basex + total_flow_int[:,:,0]
        numpy.clip(ix, 0, self.width - 1, ix)
        return bitmap[iy, ix]

    def get_accumulator_array(self) -> numpy.ndarray:
        return numpy.copy(self.total_flow)
