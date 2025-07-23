import enum
import os
import re
import warnings

import numpy

from .accumulator import Accumulator
from ..utils import parse_hex_color, load_image, load_mask
from ..flow import FlowDirection


class CanvasAccumulator(Accumulator):

    @enum.unique
    class BitmapIntroductionFlags(enum.Enum):
        MOTION = 1
        STATIC = 2
        NO_OVERWRITE = 4
        OUTER_FILL = 8

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
        if self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.OUTER_FILL.value:
            self.mask = numpy.ones((self.height, self.width), dtype=numpy.uint8)
            self.outer_mask = numpy.zeros((self.height, self.width), dtype=numpy.uint8)
        if self.reset_mode not in [Accumulator.ResetMode.OFF, Accumulator.ResetMode.RANDOM]:
            warnings.warn(f"Unsupported reset mode '{self.reset_mode}' for CanvasAccumulator")

    def update(self, flow: numpy.ndarray, direction: FlowDirection):
        self._update_flow(flow)
        self.direction = direction
        if self.reset_mode == Accumulator.ResetMode.RANDOM:
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
            if self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.NO_OVERWRITE.value:
                wh = numpy.where(self.mask.flat[bitmap_target] == 0)
                bitmap_source = bitmap_source[wh]
                bitmap_target = bitmap_target[wh]

        elif self.direction == FlowDirection.BACKWARD:
            shift = numpy.arange(self.height * self.width) + self.flow_flat
            new_mask = self.mask.flat[shift].reshape((self.height, self.width))
            pixels_target = numpy.nonzero(new_mask.flat)[0]
            pixels_source = pixels_target + self.flow_flat[pixels_target]
            new_bitmap_mask = bitmap_mask.flat[shift].reshape((self.height, self.width))
            if self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.NO_OVERWRITE.value:
                bitmap_target = numpy.nonzero(numpy.multiply(numpy.multiply(new_bitmap_mask.flat, self.flow_flat), (1 - self.mask).flat))[0]
            else:
                bitmap_target = numpy.nonzero(numpy.multiply(new_bitmap_mask.flat, self.flow_flat))[0]
            bitmap_source = bitmap_target + self.flow_flat[bitmap_target]
            
        else:
            raise ValueError(f"Unknown flow direction {self.direction}")

        aux = self.canvas.copy()

        if self.crumble:
            self.put(pixels_source, pixels_source, self.initial_canvas)
            self.mask.flat[pixels_source] = 0

        self.put(pixels_target, pixels_source, aux)
        self.mask.flat[pixels_target] = 1

        if self.outer_mask is not None:
            self.outer_mask.flat[pixels_target] = self.outer_mask.flat[pixels_source]

        if self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.STATIC.value:
            static_bitmap_source = numpy.nonzero(bitmap_mask.flat)[0]
            self.put(static_bitmap_source, static_bitmap_source, bitmap)
            self.mask.flat[static_bitmap_source] = 1
            if self.outer_mask is not None:
                self.outer_mask.flat[static_bitmap_source] = 1

        if self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.MOTION.value:
            self.put(bitmap_target, bitmap_source, bitmap)
            self.mask.flat[bitmap_target] = 1
            if self.outer_mask is not None:
                self.outer_mask.flat[bitmap_target] = 1

        if self.outer_mask is not None and self.bitmap_introduction_flags & CanvasAccumulator.BitmapIntroductionFlags.OUTER_FILL.value:
            outer_mask_inds = numpy.nonzero(self.outer_mask.flat)[0]
            self.put(outer_mask_inds, outer_mask_inds, bitmap)

        return self.canvas

    def get_accumulator_array(self) -> numpy.ndarray:
        a = numpy.zeros((self.height, self.width, 2))
        a[:,:,0] = self.mask
        a[:,:,1] = self.mask
        return a