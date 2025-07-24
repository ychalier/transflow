import numpy

from .accumulator import Accumulator
from ..flow import Direction


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

    def update(self, flow: numpy.ndarray, direction: Direction):
        self._update_flow(flow)
        if self.reset_mode == Accumulator.ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap <= self.heatmap_reset_threshold)
                & (numpy.random.random(size=(self.height, self.width)) <= threshold))
            self.mapx[where] = self.basex[where]
            self.mapy[where] = self.basey[where]
        elif self.reset_mode == Accumulator.ResetMode.LINEAR:
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
        if direction == Direction.FORWARD:
            where = numpy.nonzero(self.flow_flat)
            numpy.put(self.mapx, self.base_flat[where] + self.flow_flat[where],
                      self.mapx.flat[where], mode="clip")
            numpy.put(self.mapy, self.base_flat[where] + self.flow_flat[where],
                      self.mapy.flat[where], mode="clip")
        elif direction == Direction.BACKWARD:
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
