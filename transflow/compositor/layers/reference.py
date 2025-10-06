import enum

import numpy

from transflow.compositor.pixmap_source_interface import PixmapSourceInterface

from ...types import Flow, FloatMask, BoolMask
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

    INDEX_SOURCE: int = 3

    def __init__(self, *args):
        DataLayer.__init__(self, *args)
        self.data[:,:,0:2] = self.base.copy() # shape: (height, width, 2) [i, j]
        self.data[:,:,self.INDEX_ALPHA] = 1
        self._set_base_source_indices()
        self.reset_mode: ResetMode = ResetMode.from_string(self.config.reset_mode)
        self.reset_mask: FloatMask = load_float_mask(self.config.reset_mask, (self.height, self.width), 1)

    def _set_base_source_indices(self, mask: BoolMask | None = None):
        for i, source in enumerate(self.sources):
            if mask is None:
                where = numpy.nonzero(source.introduction_mask)
            else:
                where = numpy.nonzero(numpy.multiply(source.introduction_mask, mask))
            self.data[:,:,self.INDEX_SOURCE][where] = i

    def set_sources(self, sources: list[PixmapSourceInterface]):
        DataLayer.set_sources(self, sources)
        self._set_base_source_indices()

    def _update_reset_random(self):
        random = numpy.random.random(size=(self.height, self.width))
        reset_mask = numpy.zeros((self.height, self.width), dtype=numpy.bool)
        reset_mask[numpy.where(random < self.config.reset_random_factor * self.reset_mask)] = 1
        where = numpy.nonzero(reset_mask)
        self.data[:,:,self.INDEX_I][where] = self.base[:,:,0][where]
        self.data[:,:,self.INDEX_J][where] = self.base[:,:,1][where]
        self.data[:,:,self.INDEX_ALPHA][where] = 1
        if self.config.reset_source:
            self._set_base_source_indices(reset_mask)

    def _update_reset_constant(self):
        dij_base = (self.base - self.data[:,:,(self.INDEX_I, self.INDEX_J)]).astype(numpy.float32)
        dij = dij_base.copy()
        norm_base = numpy.linalg.norm(dij, ord=float("inf"), axis=2)
        where = numpy.nonzero(norm_base)
        dij[where] /= norm_base.reshape((self.height, self.width, 1))[where]
        dij *= self.config.reset_constant_step * self.reset_mask.reshape((self.height, self.width, 1))
        norm_scaled = numpy.linalg.norm(dij, ord=float("inf"), axis=2)
        where = numpy.where(norm_scaled > norm_base)
        dij[where] = dij_base[where]
        self.data[:,:,[self.INDEX_I, self.INDEX_J]] += numpy.round(dij).astype(numpy.int32)

    def _update_reset_linear(self):
        dij = self.config.reset_linear_factor * (self.base - self.data[:,:,(self.INDEX_I, self.INDEX_J)])
        self.data[:,:,[self.INDEX_I, self.INDEX_J]] += numpy.round(numpy.multiply(self.reset_mask.reshape((self.height, self.width, 1)), dij)).astype(numpy.int32)

    def _update_reset(self):
        if self.reset_mode == ResetMode.RANDOM:
            self._update_reset_random()
        elif self.reset_mode == ResetMode.CONSTANT:
            self._update_reset_constant()
        elif self.reset_mode == ResetMode.LINEAR:
            self._update_reset_linear()

    def _update_rgba(self):
        for i, source in enumerate(self.sources):
            where = numpy.ones((self.height, self.width), dtype=numpy.bool)
            where[numpy.where(self.data[:,:,self.INDEX_SOURCE] != i)] = 0
            where[numpy.where(self.data[:,:,self.INDEX_ALPHA] == 0)] = 0
            where = numpy.nonzero(where)
            pixmap = source.next()
            mapping_i = numpy.clip(numpy.round(self.data[:,:,0]), 0, self.height - 1)[where]
            mapping_j = numpy.clip(numpy.round(self.data[:,:,1]), 0, self.width - 1)[where]
            self.rgba[:,:,:pixmap.shape[2]][where] = pixmap[mapping_i, mapping_j]
            if pixmap.shape[2] == 3:
                self.rgba[:,:,3] = 0
                self.rgba[:,:,3][where] = 1

    def update(self, flow: Flow):
        self._update_reset()
        self._update_rgba()
