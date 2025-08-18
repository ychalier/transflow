import math
import os
import random
import re
import warnings
from typing import Callable

import numpy


def load_image(image_path: str) -> numpy.ndarray:
    import PIL.Image
    image = PIL.Image.open(image_path)
    arr = numpy.array(image)
    image.close()
    return arr


def load_mask(mask_path: str, newaxis: bool = False) -> numpy.ndarray:
    arr = load_image(mask_path).astype(numpy.float32)
    if arr.ndim == 2:
        mask = arr / 255
    elif arr.ndim == 3:
        mask = numpy.mean(arr[:,:,:3], axis=2) / 255
    else:
        raise ValueError(f"Image has wrong number of dimensions {arr.ndim}, expected 2 or 3")
    if newaxis:
        return mask.reshape((*mask.shape, 1))
    return mask


def find_unique_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if root.endswith(".flow") or root.endswith(".map"):
        root, pre_ext = os.path.splitext(root)
        ext = pre_ext + ext
    i = 0
    m = re.match(r".*\.(\d{3})$", root)
    if m:
        i = int(m.group(1)) + 1
        root = root[:-4]
    while os.path.isfile(path):
        path = root + f".{i:03d}" + ext
        i += 1
    return path


def parse_hex_color(string: str) -> tuple[int, int, int]:
    s = string.replace("#", "").replace("0x", "").replace("x", "")
    x = int(s, 16)
    return ((x >> 16) & 255, (x >> 8) & 255, x & 255)


def compose_top(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    return colors[-1]


def compose_additive(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        min(255, sum(c[0] for c in colors)),
        min(255, sum(c[1] for c in colors)),
        min(255, sum(c[2] for c in colors))
    )


def compose_subtractive(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = colors[0]
    for color in colors[1:]:
        r = max(0, r - (255 - color[0]))
        g = max(0, g - (255 - color[1]))
        b = max(0, b - (255 - color[2]))
    return (r, g, b)


def compose_average(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    n = len(colors)
    if n == 0:
        return (0, 0, 0)
    return (
        int(sum(c[0] for c in colors) / n),
        int(sum(c[1] for c in colors) / n),
        int(sum(c[2] for c in colors) / n),
    )


def multiply_arrays(arrays: list[numpy.ndarray]) -> numpy.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    out = numpy.multiply(arrays[0], arrays[1])
    for array in arrays[2:]:
        numpy.multiply(out, array, out)
    return out


def binarize_arrays(arrays: list[numpy.ndarray]) -> list[numpy.ndarray]:
    for array in arrays:
        where = numpy.where(numpy.abs(array) > 0.2)
        array[:,:] = 0
        array[where] = 1
    return arrays


def absmax(arrays: list[numpy.ndarray]) -> numpy.ndarray:
    w, h = arrays[0].shape[0:2]
    stack = numpy.stack(arrays).reshape((2, w * h * 2))
    abs_stack = numpy.abs(stack)
    argmax = numpy.argmax(abs_stack, axis=0).reshape((1, w * h * 2))
    return numpy.take_along_axis(stack, argmax, 0).reshape(arrays[0].shape)


def startfile(path: str):
    import sys, subprocess
    if sys.platform == "win32":
        os.startfile(os.path.realpath(path))
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, os.path.realpath(path)])


def parse_timestamp(timestamp: str | float | int | None) -> float | None:
    if timestamp is None or isinstance(timestamp, float) or isinstance(timestamp, int):
        return timestamp
    a = re.match(r"(\d\d):(\d\d):(\d\d)(?:\.(\d\d\d))?", timestamp)
    if a is None:
        warnings.warn(f"Could not parse timestamp {timestamp}")
        return None
    h = int(a.group(1))
    m = int(a.group(2))
    s = int(a.group(3))
    ms = 0
    if a.group(4) is not None:
        ms = int(a.group(4))
    return 3600 * h + 60 * m + s + ms/1000


def parse_lambda_expression(expr_string: str, vars: tuple[str, ...] = ("t", )) -> Callable:
    if len(vars) == 1:
        vars_str = vars[0]
    else:
        vars_str = ",".join(vars)
    return eval(f"lambda {vars_str}: " + expr_string)


def upscale_array(arr: numpy.ndarray, wf: int, hf: int) -> numpy.ndarray:
    return numpy.kron(arr * (wf, hf), numpy.ones((hf, wf, 1))).astype(arr.dtype)


def putn(target_array: numpy.ndarray, source_array: numpy.ndarray, target_inds: numpy.ndarray, source_inds: numpy.ndarray, scale: int):
    target_inds_scaled = target_inds * scale
    source_inds_scaled = source_inds * scale
    for i in range(scale):
        target_array.flat[target_inds_scaled + i] = source_array.flat[source_inds_scaled + i]


def putn_1d(target_array: numpy.ndarray, value: int | float, target_inds: numpy.ndarray, scale: int, axis: int):
    target_inds_scaled = target_inds * scale
    target_array.flat[target_inds_scaled + axis] = value
