import numpy

from .layer import Layer


class DataLayer(Layer):

    DEPTH: int = 4
    INDEX_I: int = 0
    INDEX_J: int = 1
    INDEX_ALPHA: int = 2
    INDEX_SOURCE: int = 3

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.base = numpy.indices((self.height, self.width), dtype=numpy.int32).transpose(1, 2, 0)
        self.data = numpy.zeros((self.height, self.width, self.DEPTH), dtype=numpy.int32)