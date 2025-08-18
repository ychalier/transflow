import numpy

from ...types import Flow
from .reference import ReferenceLayer


class SumLayer(ReferenceLayer):

    def _update_sum(self, flow: Flow):
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += numpy.floor(flow).astype(numpy.int32)

    def update(self, flow: Flow):
        self._update_sum(flow)
        ReferenceLayer.update(self, flow)

