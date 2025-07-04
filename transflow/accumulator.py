import enum
import logging
import os
import re
import warnings

import numpy

from .flow import FlowDirection
from .utils import parse_hex_color, compose_top, compose_additive,\
    compose_subtractive, compose_average, load_image, load_mask


red = numpy.zeros((1080, 1920, 3))
red[:,:,0] = 255
green = numpy.zeros((1080, 1920, 3))
green[:,:,1] = 255


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

    def __init__(self,
            width: int,
            height: int,
            heatmap_mode: HeatmapMode | str = "discrete",
            heatmap_args: str | tuple[int|float] = "0:0:0:0",
            heatmap_reset_threshold: float | None = None,
            reset_mode: ResetMode | str = "off",
            reset_alpha: float = .9,
            reset_mask_path: str | None = None):
        self.width = width
        self.height = height
        self.flow_int: numpy.ndarray = None
        self.flow_flat: numpy.ndarray = None
        self.heatmap_mode = HeatmapMode.from_string(heatmap_mode)\
            if isinstance(heatmap_mode, str) else heatmap_mode
        self.heatmap_reset_threshold = heatmap_reset_threshold if heatmap_reset_threshold is not None else float("inf")
        if self.heatmap_mode == HeatmapMode.DISCRETE:
            self.heatmap = numpy.zeros((self.height, self.width), dtype=numpy.int32)
            x = tuple(map(int, heatmap_args.split(":")))\
                if isinstance(heatmap_args, str) else heatmap_args
            self.heatmap_min = x[0]
            self.heatmap_max = x[1]
            self.heatmap_add = x[2]
            self.heatmap_sub = x[3]
        elif self.heatmap_mode == HeatmapMode.CONTINUOUS:
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
        self.reset_mode = reset_mode if isinstance (reset_mode, ResetMode) else ResetMode.from_string(reset_mode)
        self.reset_alpha = reset_alpha
        self.reset_mask = None if reset_mask_path is None else load_mask(reset_mask_path)

    def _update_flow(self, flow: numpy.ndarray):
        numpy.clip(flow[:,:,0], self.fx_min, self.fx_max, flow[:,:,0])
        numpy.clip(flow[:,:,1], self.fy_min, self.fy_max, flow[:,:,1])
        self.flow_int = numpy.round(flow).astype(numpy.int32)
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
            reset_mode = ResetMode.from_string(reset_mode)
        match method:
            case "map":
                return MappingAccumulator(width, height, **args)
            case "stack":
                return StackAccumulator(
                    width, height,
                    bg_color,
                    stack_composer,
                    **args)
            case "sum":
                return SumAccumulator(width, height, **args)
            case "crumble":
                return CrumbleAccumulator(width, height, bg_color, **args)
            case "canvas":
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


class MappingAccumulator(Accumulator):

    def __init__(self, width: int, height: int, **acc_args):
        Accumulator.__init__(self, width, height, **acc_args)
        self.base_flat = numpy.arange(self.height * self.width)
        assert self.base_flat.dtype == int, self.base_flat.dtype
        shape = (self.height, self.width)
        self.basex = numpy.broadcast_to(numpy.arange(self.width), shape).copy()
        self.basey = numpy.broadcast_to(numpy.arange(self.height)[:,numpy.newaxis], shape).copy()
        self.mapx = self.basex.astype(numpy.float32)
        self.mapy = self.basey.astype(numpy.float32)

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap <= self.heatmap_reset_threshold)
                & (numpy.random.random(size=(self.height, self.width)) <= threshold))
            self.mapx[where] = self.basex[where]
            self.mapy[where] = self.basey[where]
        elif self.reset_mode == ResetMode.LINEAR:
            where = numpy.where(self.heatmap <= self.heatmap_reset_threshold)
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
        mapping = numpy.clip(self.mapy.astype(numpy.int32), 0, self.height - 1) * self.width\
            + numpy.clip(self.mapx.astype(numpy.int32), 0, self.width - 1)
        out = bitmap\
            .reshape((self.height * self.width, bitmap.shape[2]))[mapping.flat]\
            .reshape(bitmap.shape)
        return out

    def get_accumulator_array(self) -> numpy.ndarray:
        return numpy.stack([self.mapx - self.basex, self.mapy - self.basey], axis=-1)


class CrumbleAccumulator(MappingAccumulator):

    def __init__(self, width: int, height: int, bg_color: str = "000000", **acc_args):
        MappingAccumulator.__init__(self, width, height, **acc_args)
        if self.reset_mode not in [ResetMode.OFF, ResetMode.RANDOM]:
            warnings.warn(
                f"CrumbleAccumulator only works with Off or Random reset, not {self.reset_mode}")
        self.bg_color = parse_hex_color(bg_color)
        self.crumble_mask = numpy.ones((self.height, self.width), dtype=numpy.uint8)

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap <= self.heatmap_reset_threshold)
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
        mapping = numpy.clip(self.mapy.astype(numpy.int32), 0, self.height - 1) * self.width\
            + numpy.clip(self.mapx.astype(numpy.int32), 0, self.width - 1)
        out = bitmap\
            .reshape((self.height * self.width, bitmap.shape[2]))[mapping.flat]\
            .reshape(bitmap.shape)
        out[self.crumble_mask == 0] = self.bg_color
        return out


class StackAccumulator(Accumulator):

    def __init__(self, width: int, height: int, bg_color: str = "ffffff",
                 composer: str = "top", **acc_args):
        Accumulator.__init__(self, width, height, **acc_args)
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

    def __init__(self, width: int, height: int, **acc_args):
        Accumulator.__init__(self, width, height, **acc_args)
        self.total_flow = numpy.zeros((height, width, 2), dtype=numpy.float32)
        shape = (height, width)
        self.basex = numpy.broadcast_to(numpy.arange(self.width), shape).copy()
        self.basey = numpy.broadcast_to(numpy.arange(self.height)[:,numpy.newaxis], shape).copy()

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        if direction != FlowDirection.BACKWARD:
            warnings.warn(f"SumAccumulator only works with backward flow, not {direction}")
        self._update_flow(flow)
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            below_threshold = numpy.random.random(size=(self.height, self.width)) <= threshold
            below_heatmap = self.heatmap <= self.heatmap_reset_threshold
            where = numpy.nonzero(numpy.multiply(below_threshold, below_heatmap))
            self.total_flow[where] = 0
        elif self.reset_mode == ResetMode.LINEAR:
            wh = numpy.where(self.heatmap <= self.heatmap_reset_threshold)
            if self.reset_mask is None:
                self.total_flow[wh] = (1 - self.reset_alpha) * self.total_flow[wh]
            else:
                self.total_flow[wh] = numpy.expand_dims(1 - self.reset_mask[wh], -1) * self.total_flow[wh]
        self.total_flow += flow

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        total_flow_int = numpy.round(self.total_flow).astype(numpy.int32)
        iy = self.basey + total_flow_int[:,:,1]
        numpy.clip(iy, 0, self.height - 1, iy)
        ix = self.basex + total_flow_int[:,:,0]
        numpy.clip(ix, 0, self.width - 1, ix)
        return bitmap[iy, ix]

    def get_accumulator_array(self) -> numpy.ndarray:
        return numpy.copy(self.total_flow)


@enum.unique
class BitmapIntroductionFlags(enum.Enum):
    MOTION = 1
    STATIC = 2
    NO_OVERWRITE = 4
    OUTER_FILL = 8


class CanvasAccumulator(Accumulator):

    def __init__(self,
            width: int,
            height: int,
            initial_canvas: str | None = None,
            bitmap_mask_path: str | None = None,
            crumble: bool = False,
            bitmap_introduction_flags: int = 1,
            **acc_args):
        Accumulator.__init__(self, width, height, **acc_args)
        self.initial_canvas = 255 * numpy.ones((height, width, 3), dtype=numpy.uint8)
        if initial_canvas is not None:
            if re.match(r"#?[a-f0-9]{6}", initial_canvas):
                self.initial_canvas[:,:] = parse_hex_color(initial_canvas)
            elif os.path.isfile(initial_canvas):
                self.initial_canvas = load_image(initial_canvas)[:,:,:3]
            else:
                warnings.warn(f"Could not use inital canvas argument {initial_canvas}")
        self.canvas = self.initial_canvas.copy()
        self.bitmap_mask = None if bitmap_mask_path is None else load_mask(bitmap_mask_path)
        self.direction: FlowDirection | None = None
        self.crumble = crumble
        self.mask = numpy.zeros((self.height, self.width), dtype=numpy.uint8)
        self.outer_mask = None
        self.bitmap_introduction_flags = bitmap_introduction_flags
        if self.bitmap_introduction_flags & BitmapIntroductionFlags.OUTER_FILL.value:
            self.mask = numpy.ones((self.height, self.width), dtype=numpy.uint8)
            self.outer_mask = numpy.zeros((self.height, self.width), dtype=numpy.uint8)
        if self.reset_mode not in [ResetMode.OFF, ResetMode.RANDOM]:
            warnings.warn(f"Unsupported reset mode '{self.reset_mode}' for CanvasAccumulator")

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        self.direction = direction
        if self.reset_mode == ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            below_threshold = numpy.random.random(size=(self.height, self.width)) <= threshold
            below_heatmap = self.heatmap <= self.heatmap_reset_threshold
            where = numpy.nonzero(numpy.multiply(below_threshold, below_heatmap))
            self.canvas[where] = self.initial_canvas[where]
            self.mask[where] = 0

    def put(self, target: numpy.ndarray, source: numpy.ndarray, values: numpy.ndarray):
        t3 = target * 3
        s3 = source * 3
        self.canvas.flat[t3] = values.flat[s3]
        self.canvas.flat[t3+1] = values.flat[s3+1]
        self.canvas.flat[t3+2] = values.flat[s3+2]

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:

        bitmap_mask = numpy.ones((self.height, self.width), dtype=numpy.uint8) if self.bitmap_mask is None else self.bitmap_mask

        if self.direction == FlowDirection.FORWARD:
            pixels_source = numpy.nonzero(self.mask.flat)[0]
            pixels_target = pixels_source + self.flow_flat[pixels_source]
            bitmap_source = numpy.nonzero(numpy.multiply(bitmap_mask.flat, self.flow_flat))[0]
            bitmap_target = bitmap_source + self.flow_flat[bitmap_source]
            if self.bitmap_introduction_flags & BitmapIntroductionFlags.NO_OVERWRITE.value:
                wh = numpy.where(self.mask.flat[bitmap_target] == 0)
                bitmap_source = bitmap_source[wh]
                bitmap_target = bitmap_target[wh]

        elif self.direction == FlowDirection.BACKWARD:
            shift = numpy.arange(self.height * self.width) + self.flow_flat
            new_mask = self.mask.flat[shift].reshape((self.height, self.width))
            pixels_target = numpy.nonzero(new_mask.flat)[0]
            pixels_source = pixels_target + self.flow_flat[pixels_target]
            new_bitmap_mask = bitmap_mask.flat[shift].reshape((self.height, self.width))
            if self.bitmap_introduction_flags & BitmapIntroductionFlags.NO_OVERWRITE.value:
                bitmap_target = numpy.nonzero(numpy.multiply(numpy.multiply(new_bitmap_mask.flat, self.flow_flat), (1 - self.mask).flat))[0]
            else:
                bitmap_target = numpy.nonzero(numpy.multiply(new_bitmap_mask.flat, self.flow_flat))[0]
            bitmap_source = bitmap_target + self.flow_flat[bitmap_target]

        aux = self.canvas.copy()

        if self.crumble:
            self.put(pixels_source, pixels_source, self.initial_canvas)
            self.mask.flat[pixels_source] = 0

        self.put(pixels_target, pixels_source, aux)
        self.mask.flat[pixels_target] = 1

        if self.outer_mask is not None:
            self.outer_mask.flat[pixels_target] = self.outer_mask.flat[pixels_source]

        if self.bitmap_introduction_flags & BitmapIntroductionFlags.STATIC.value:
            static_bitmap_source = numpy.nonzero(bitmap_mask.flat)[0]
            self.put(static_bitmap_source, static_bitmap_source, bitmap)
            self.mask.flat[static_bitmap_source] = 1
            if self.outer_mask is not None:
                self.outer_mask.flat[static_bitmap_source] = 1

        if self.bitmap_introduction_flags & BitmapIntroductionFlags.MOTION.value:
            self.put(bitmap_target, bitmap_source, bitmap)
            self.mask.flat[bitmap_target] = 1
            if self.outer_mask is not None:
                self.outer_mask.flat[bitmap_target] = 1

        if self.outer_mask is not None and self.bitmap_introduction_flags & BitmapIntroductionFlags.OUTER_FILL.value:
            outer_mask_inds = numpy.nonzero(self.outer_mask.flat)[0]
            self.put(outer_mask_inds, outer_mask_inds, bitmap)

        return self.canvas

    def get_accumulator_array(self) -> numpy.ndarray:
        a = numpy.zeros((self.height, self.width, 2))
        a[:,:,0] = self.mask
        a[:,:,1] = self.mask
        return a