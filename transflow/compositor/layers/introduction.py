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
        # self.introduction_masks: list[BoolMask] = []
        # for _ in self.sources:
        #     self.introduction_masks.append(numpy.ones((self.height, self.width), dtype=numpy.bool))
        self.introduced_once: bool = False

    def _update_introduction(self):
        if self.config.introduce_once and self.introduced_once:
            return
        self.introduced_once = True
        mask = numpy.ones((self.height, self.width), dtype=numpy.bool)
        if not self.config.introduce_pixels_on_empty_spots:
            mask[numpy.where(self.data[:,:,3]) == 0] = 0
        if not self.config.introduce_pixels_on_filled_spots:
            mask[numpy.nonzero(self.data[:,:,3])] = 0
        if not self.config.introduce_moving_pixels:
            mask.flat[numpy.nonzero(self.flow_flat)] = 0
        if not self.config.introduce_unmoving_pixels:
            mask.flat[numpy.where(self.flow_flat) == 0] = 0
        for i, source in enumerate(self.sources):
            pixmap = source.next()
            where_target = numpy.nonzero(numpy.multiply(mask.flat, self.mask_introduction.flat))[0]
            where_source = where_target + self.flow_flat[where_target]
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
