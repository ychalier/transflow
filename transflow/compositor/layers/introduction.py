import numpy

from ...utils import load_bool_mask, putn
from ...types import Flow, BoolMask
from .movement import MovementLayer


class IntroductionLayer(MovementLayer):

    DEPTH = 8 # r, g, b, alpha, source, i, j, frame
    POS_I_IDX: int = 5
    POS_J_IDX: int = 6
    POS_A_IDX: int = 3

    def __init__(self, *args):
        MovementLayer.__init__(self, *args)
        # TODO: consider using one mask per source
        self.mask_introduction: BoolMask = load_bool_mask(self.config.mask_introduction, (self.height, self.width), True)
        self.introduced_once: bool = False

    def _update_introduction(self):
        if self.config.introduce_once and self.introduced_once:
            return
        self.introduced_once = True
        mask = numpy.ones((self.height, self.width), dtype=numpy.bool)

        where_empty = numpy.where(self.data[:,:,self.POS_A_IDX]) == 0
        where_filled = numpy.nonzero(self.data[:,:,self.POS_A_IDX])

        if not self.config.introduce_pixels_on_empty_spots:
            mask[where_empty] = 0
        if not self.config.introduce_pixels_on_filled_spots:
            mask[where_filled] = 0
        if not self.config.introduce_moving_pixels:
            mask.flat[numpy.nonzero(self.flow_flat)] = 0
        if not self.config.introduce_unmoving_pixels:
            mask.flat[numpy.where(self.flow_flat) == 0] = 0

        mask = numpy.multiply(mask, self.mask_introduction)

        consider_flow = not (self.config.introduce_on_all_filled_spots or self.config.introduce_on_all_empty_spots)
        if self.config.introduce_on_all_filled_spots:
            mask[where_filled] = 1
        if self.config.introduce_on_all_empty_spots:
            mask[where_empty] = 1

        for i, source in enumerate(self.sources):
            pixmap = source.next()
            where_target = numpy.nonzero(mask.flat)[0]
            if consider_flow:
                where_source = where_target + self.flow_flat[where_target]
            else:
                where_source = where_target
            arrays = [
                pixmap,
                numpy.broadcast_to(i, (self.height, self.width, 1)),
                self.base,
                numpy.broadcast_to(source.frame_number, (self.height, self.width, 1)),
            ]
            if pixmap.shape[2] == 3:
                arrays.insert(1, numpy.broadcast_to(1, (self.height, self.width, 1)))
            putn(self.data, numpy.concat(arrays, axis=2), where_target, where_source, 8)

    def _update_rgba(self):
        self.rgba = self.data[:,:,:4]

    def update(self, flow: Flow):
        MovementLayer.update(self, flow)
        self._update_introduction()
        self._update_rgba()
