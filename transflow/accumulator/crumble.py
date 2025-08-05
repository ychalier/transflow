import warnings

import numpy

from .accumulator import Accumulator
from .mapping import MappingAccumulator
from ..utils import parse_hex_color


class CrumbleAccumulator(MappingAccumulator):

    def __init__(self, width: int, height: int, bg_color: str = "000000", **acc_args):
        MappingAccumulator.__init__(self, width, height, **acc_args)
        if self.reset_mode not in [Accumulator.ResetMode.OFF, Accumulator.ResetMode.RANDOM]:
            warnings.warn(
                f"CrumbleAccumulator only works with Off or Random reset, not {self.reset_mode}")
        self.bg_color = parse_hex_color(bg_color)
        self.crumble_mask = numpy.ones((self.height, self.width), dtype=numpy.uint8)

    def update(self, flow: numpy.ndarray):
        self._update_flow(flow)
        if self.reset_mode == Accumulator.ResetMode.RANDOM:
            threshold = self.reset_alpha if self.reset_mask is None else self.reset_mask
            where = numpy.where(
                (self.heatmap <= self.heatmap_reset_threshold)
                & (numpy.random.random(size=(self.height, self.width)) <= threshold))
            self.mapx[where] = self.basex[where]
            self.mapy[where] = self.basey[where]
            self.crumble_mask[where] = 1
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