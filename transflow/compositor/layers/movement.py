from typing import cast

import numpy

from ...utils import load_bool_mask, putn, putn_1d
from ...types import Flow, BoolMask
from .data import DataLayer


class MovementLayer(DataLayer):

    def __init__(self, *args):
        DataLayer.__init__(self, *args)
        self.mask_src: BoolMask = load_bool_mask(self.config.mask_src, (self.height, self.width), True)
        self.mask_dst: BoolMask = load_bool_mask(self.config.mask_dst, (self.height, self.width), True)
        self.flow: Flow = cast(Flow, numpy.zeros((self.height, self.width, 2), dtype=numpy.float32))
        self.flow_int = numpy.zeros((self.height, self.width, 2), dtype=numpy.int32)
        self.flow_flat = numpy.zeros((self.height * self.width, 1), dtype=numpy.int32)

    def _update_flow(self, flow: Flow):
        self.flow = flow
        self.flow_int = numpy.round(self.flow).astype(numpy.int32)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])

    def _update_move(self):
        shift = numpy.arange(self.height * self.width) + self.flow_flat
        mask_src = self.mask_src.copy()
        
        mask_src_filled = None
        if self.config.transparent_pixels_can_move:
            mask_src_filled = numpy.ones((self.height, self.width), dtype=numpy.bool)
            mask_src_filled[numpy.where(self.data[:,:,self.INDEX_ALPHA] == 0)] = 0
            mask_src_filled = mask_src_filled.flat[shift].reshape((self.height, self.width))
        else:
            mask_src[numpy.where(self.data[:,:,self.INDEX_ALPHA] == 0)] = 0

        if not self.config.transparent_pixels_can_move:
            mask_src[numpy.where(self.data[:,:,self.INDEX_ALPHA] == 0)] = 0
        mask_src = mask_src.flat[shift].reshape((self.height, self.width))

        mask_dst = self.mask_dst.copy()
        if not self.config.pixels_can_move_to_empty_spot:
            mask_dst[numpy.where(self.data[:,:,self.INDEX_ALPHA] == 0)] = 0
        if not self.config.pixels_can_move_to_filled_spot:
            mask_dst[numpy.nonzero(self.data[:,:,self.INDEX_ALPHA])] = 0

        mask_all_flat = numpy.multiply(mask_src.flat, mask_dst.flat)
        where_target = numpy.nonzero(numpy.multiply(self.flow_flat, mask_all_flat))[0]
        where_source = where_target + self.flow_flat[where_target]

        aux = self.data.copy()
        putn(self.data, aux, where_target, where_source, self.DEPTH)
        if self.config.moving_pixels_leave_empty_spot:
            putn_1d(self.data[:,:,self.INDEX_ALPHA], 0, where_source, 1, 0)
        if self.config.transparent_pixels_can_move:
            assert mask_src_filled is not None
            where_target_filled = numpy.nonzero(numpy.multiply(self.flow_flat, numpy.multiply(mask_all_flat, mask_src_filled.flat)))[0]
            putn_1d(self.data[:,:,self.INDEX_ALPHA], 1, where_target_filled, 1, 0)
        else:
            putn_1d(self.data[:,:,self.INDEX_ALPHA], 1, where_target, 1, 0)

    def update(self, flow: Flow):
        self._update_flow(flow)
        self._update_move()
