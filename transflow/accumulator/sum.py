import warnings

import numpy

from .accumulator import Accumulator
from ..flow import FlowDirection


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
        if self.reset_mode == Accumulator.ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            below_threshold = numpy.random.random(size=(self.height, self.width)) <= threshold
            below_heatmap = self.heatmap <= self.heatmap_reset_threshold
            where = numpy.nonzero(numpy.multiply(below_threshold, below_heatmap))
            self.total_flow[where] = 0
        elif self.reset_mode == Accumulator.ResetMode.LINEAR:
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