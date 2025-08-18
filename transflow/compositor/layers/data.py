import numpy

from .layer import Layer


class DataLayer(Layer):

    DEPTH: int = 4
    POS_I_IDX: int = 0
    POS_J_IDX: int = 1
    POS_A_IDX: int = 2

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.base = numpy.indices((self.height, self.width), dtype=numpy.int32).transpose(1, 2, 0)
        self.data = numpy.zeros((self.height, self.width, self.DEPTH), dtype=numpy.int32)