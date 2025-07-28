import numpy

from ..utils import parse_hex_color


def render1d(
        arr: numpy.ndarray,
        scale: float = 1,
        colors: tuple[str, ...] | None = None,
        binary: bool = False
        ) -> numpy.ndarray:
    if colors is None:
        colors = ("#000000", "#ffffff")
    color_arrs = [numpy.array(parse_hex_color(c), dtype=numpy.float32) for c in colors]
    out_shape = (*arr.shape[:2], 1)
    if binary:
        coeff = numpy.clip(numpy.round(scale * arr), 0, 1).reshape(out_shape)
        coeff_a = 1 - coeff
        coeff_b = coeff
    else:
        coeff_a = numpy.clip(1 - scale * arr, 0, 1).reshape(out_shape)
        coeff_b = numpy.clip(scale * arr, 0, 1).reshape(out_shape)
    frame = numpy.multiply(coeff_a, color_arrs[0]) + numpy.multiply(coeff_b, color_arrs[1])
    return numpy.clip(frame, 0, 255).astype(numpy.uint8)


def render2d(
        arr: numpy.ndarray,
        scale: float = 1,
        colors: tuple[str, ...] | None = None
        ) -> numpy.ndarray:
    if colors is None:
        colors = ("#ffff00", "#0000ff", "#ff00ff", "#00ff00")
    color_arrs = [numpy.array(parse_hex_color(c), dtype=numpy.float32) for c in colors]
    out_shape = (*arr.shape[:2], 1)
    coeff_y = numpy.clip(1 + scale * arr[:,:,0], 0, 1).reshape(out_shape)
    coeff_b = numpy.clip(1 - scale * arr[:,:,0], 0, 1).reshape(out_shape)
    coeff_m = numpy.clip(1 + scale * arr[:,:,1], 0, 1).reshape(out_shape)
    coeff_g = numpy.clip(1 - scale * arr[:,:,1], 0, 1).reshape(out_shape)
    frame = .5 * (
        numpy.multiply(coeff_y, color_arrs[0])
        + numpy.multiply(coeff_b, color_arrs[1])
        + numpy.multiply(coeff_m, color_arrs[2])
        + numpy.multiply(coeff_g, color_arrs[3]))
    return numpy.clip(frame, 0, 255).astype(numpy.uint8)
