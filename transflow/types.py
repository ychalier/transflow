from typing import Literal, TypeVar

import numpy


Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
Grey = numpy.ndarray[tuple[Height, Width], numpy.dtype[numpy.uint8]]
Rgb = numpy.ndarray[tuple[Height, Width, Literal[3]], numpy.dtype[numpy.uint8]]
Rgba = numpy.ndarray[tuple[Height, Width, Literal[4]], numpy.dtype[numpy.uint8]]
Flow = numpy.ndarray[tuple[Height, Width, Literal[2]], numpy.dtype[numpy.float32]]
Pixmap = Rgb | Rgba
BoolMask = numpy.ndarray[tuple[Height, Width], numpy.dtype[numpy.bool]]
FloatMask = numpy.ndarray[tuple[Height, Width], numpy.dtype[numpy.float32]]
