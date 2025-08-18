from typing import Literal, TypeVar

import numpy


Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
Rgba = numpy.ndarray[tuple[Height, Width, Literal[4]], numpy.dtype[numpy.uint8]]
Flow = numpy.ndarray[tuple[Height, Width, Literal[2]], numpy.dtype[numpy.float32]]
