import enum

import numpy

from ...types import Flow, FloatMask
from ...utils import load_float_mask
from .data import DataLayer


@enum.unique
class ResetMode(enum.Enum):

    OFF = 0
    RANDOM = 1
    CONSTANT = 2
    LINEAR = 3

    @classmethod
    def from_string(cls, string):
        match string:
            case "off":
                return cls.OFF
            case "random":
                return cls.RANDOM
            case "constant":
                return cls.CONSTANT
            case "linear":
                return cls.LINEAR
        raise ValueError(f"Unknown reset mode {string}")


class ReferenceLayer(DataLayer):

    def __init__(self, *args):
        DataLayer.__init__(self, *args)
        self.data[:,:,0:2] = self.base.copy() # shape: (height, width, 2) [i, j]
        self.data[:,:,2] = 1
        self.data[:,:,3] = 0 # source index
        self.reset_mode: ResetMode = ResetMode.from_string(self.config.reset_mode)
        self.reset_mask: FloatMask = load_float_mask(self.config.reset_mask, (self.height, self.width), 1)

    def _update_reset_random(self):
        random = numpy.random.random(size=(self.height, self.width))
        reset_mask = numpy.zeros((self.height, self.width), dtype=numpy.bool)
        reset_mask[numpy.where(random < self.config.reset_random_factor * self.reset_mask)] = 1
        where = numpy.nonzero(reset_mask)
        self.data[:,:,self.POS_I_IDX][where] = self.base[:,:,0][where]
        self.data[:,:,self.POS_J_IDX][where] = self.base[:,:,1][where]
        self.data[:,:,self.POS_A_IDX][where] = 1

    def _update_reset_constant(self):
        dij = self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)]
        dij_norm = numpy.linalg.norm(dij, ord=float("inf"), axis=2)
        dij_norm[numpy.where(dij_norm > self.config.reset_constant_step)] /= self.config.reset_constant_step
        where = numpy.nonzero(dij_norm)
        dij_scaled = dij.copy()
        dij_scaled[where] = dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij_scaled

    def _update_reset_linear(self):
        dij = numpy.round(self.config.reset_linear_factor * (self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)])).astype(numpy.int32)
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij

    def _update_reset(self):
        if self.reset_mode == ResetMode.RANDOM:
            self._update_reset_random()
        elif self.reset_mode == ResetMode.CONSTANT:
            self._update_reset_constant()
        elif self.reset_mode == ResetMode.LINEAR:
            self._update_reset_linear()

    def _update_rgba(self):
        for i, source in enumerate(self.sources):
            where = numpy.where(self.data[:,:,3] == i)
            pixmap = source.next()
            mapping_i = numpy.clip(numpy.round(self.data[:,:,0]), 0, self.height - 1)[where]
            mapping_j = numpy.clip(numpy.round(self.data[:,:,1]), 0, self.width - 1)[where]
            self.rgba[:,:,:pixmap.shape[2]][where] = pixmap[mapping_i, mapping_j]
            self.rgba[:,:,3] = self.data[:,:,2] # TODO: pixmap alpha channel (if any) gets overwritten

    def update(self, flow: Flow):
        self._update_reset()
        self._update_rgba()
