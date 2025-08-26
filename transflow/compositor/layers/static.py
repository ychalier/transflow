from ...types import Flow
from .layer import Layer

import numpy


class StaticLayer(Layer):

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.rgba[:,:,3] = 1

    def update(self, flow: Flow):
        for source in self.sources:
            pixmap = source.next()
            where = numpy.nonzero(source.introduction_mask)
            self.rgba[:,:,:pixmap.shape[2]][where] = pixmap[where]
