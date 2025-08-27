import math
import os
import random
import re
import warnings
from typing import Callable, cast

import numpy

from .types import BoolMask, FloatMask


def parse_dimension_arg(arg_string: str, parent_size: int) -> int:
    if arg_string.strip() == "":
        return 0
    if arg_string.endswith("%"):
        return int(float(arg_string[:-1]) / 100 * parent_size)
    return int(arg_string)


def parse_border_args(border_string: str, height: int, width: int) -> tuple[int, int, int, int]:
    top = right = bottom = left = 0
    border_name, border_args = border_string.lower().split(":", 1)
    if border_name == "border":
        parsed_border_args = [
            parse_dimension_arg(arg_string, height if i % 2 == 0 else width)
            for i, arg_string in enumerate(border_args.split(":"))
        ]
        if len(parsed_border_args) == 1:
            top = right = bottom = left = parsed_border_args[0]
        elif len(parsed_border_args) == 2:
            top = bottom = parsed_border_args[0]
            right = left = parsed_border_args[1]
        elif len(parsed_border_args) == 4:
            top, right, bottom, left = parsed_border_args
        else:
            raise ValueError(f"Invalid number of argument {len(parsed_border_args)} for border mask")
    elif border_name == "border-top":
        top = parse_dimension_arg(border_args, height)
    elif border_name == "border-right":
        right = parse_dimension_arg(border_args, width)
    elif border_name == "border-bottom":
        bottom = parse_dimension_arg(border_args, height)
    elif border_name == "border-left":
        left = parse_dimension_arg(border_args, width)
    else:
        raise ValueError(f"Invalid border rule name {border_name}")
    return top, right, bottom, left


def load_float_mask(mask_path: str | None, shape: tuple[int, int] = (0, 0), default: float = 0) -> FloatMask:
    if mask_path is None:
        arr = numpy.zeros(shape, dtype=numpy.float32)
        arr[:,:] = default
        return arr
    inverse = False
    if mask_path is not None and mask_path.endswith(":inv"):
        inverse = True
        mask_path = mask_path[:-4]
    if mask_path.lower() == "zeros":
        arr = numpy.zeros(shape, dtype=numpy.float32)
    elif mask_path.lower() == "ones":
        arr = numpy.ones(shape, dtype=numpy.float32)
    elif mask_path.lower() == "random":
        arr = cast(FloatMask, numpy.random.rand(*shape).astype(numpy.float32))
    elif re.match(r"^border(\-(top|right|bottom|left))?:(\d+%?:|:|\d+%?$){1,4}$", mask_path, re.IGNORECASE):
        top, right, bottom, left = parse_border_args(mask_path, *shape)
        arr = numpy.zeros(shape, dtype=numpy.float32)
        if top != 0: arr[:top,:] = 1
        if right != 0: arr[:,-right:] = 1
        if bottom != 0: arr[-bottom:,:] = 1
        if left != 0: arr[:,:left] = 1
    elif re.match(r"^[hv]line:\d+%?$", mask_path, re.IGNORECASE):
        name, arg_string = mask_path.lower().split(":")
        arr = numpy.zeros(shape, dtype=numpy.float32)
        if name == "hline":
            arg = parse_dimension_arg(arg_string, shape[0])
            i = (shape[0] - arg) // 2
            arr[i:i+arg,:] = 1
        elif name == "vline":
            arg = parse_dimension_arg(arg_string, shape[1])
            j = (shape[1] - arg) // 2
            arr[:,j:j+arg] = 1
        else:
            raise ValueError(f"Invalid line rule name {name}")
    elif re.match(r"circle:\d+%?", mask_path, re.IGNORECASE):
        arg_string = mask_path.lower().split(":")[1]
        radius = parse_dimension_arg(arg_string, min(shape))
        arr = numpy.zeros(shape, dtype=numpy.float32)
        i = numpy.arange(0, shape[0])
        j = numpy.arange(0, shape[1])
        ci, cj = shape[0] // 2, shape[1] // 2
        arr = (j[numpy.newaxis,:] - cj) ** 2 + (i[:,numpy.newaxis] - ci) ** 2 < radius ** 2
    elif re.match(r"rect:\d+%?(:\d+%?)?", mask_path, re.IGNORECASE):
        arg_strings = mask_path[mask_path.index(":")+1:].split(":")
        width, height = 0, 0
        if len(arg_strings) == 1:
            width = parse_dimension_arg(arg_strings[0], shape[1])
            height = parse_dimension_arg(arg_strings[0], shape[0])
        elif len(arg_strings) == 2:
            width = parse_dimension_arg(arg_strings[0], shape[1])
            height = parse_dimension_arg(arg_strings[1], shape[0])
        else:
            raise ValueError(f"Invalid number of argument {len(arg_strings)} for rect mask")
        arr = numpy.ones(shape, dtype=numpy.float32)
        arr[:shape[0] // 2 - height // 2,:] = 0
        arr[shape[0] // 2 + height // 2:,:] = 0
        arr[:,:shape[1] // 2 - width // 2] = 0
        arr[:,shape[1] // 2 + width // 2:] = 0
    elif re.match(r"grid:\d+:\d+:\d+?", mask_path, re.IGNORECASE):
        arg_strings = mask_path[mask_path.index(":")+1:].split(":")
        nrows, ncols, radius = list(map(int, arg_strings))
        diameter = 2 * radius
        i = numpy.arange(0, diameter)
        j = numpy.arange(0, diameter)
        circle = (j[numpy.newaxis,:] - radius) ** 2 + (i[:,numpy.newaxis] - radius) ** 2 < radius ** 2
        arr = numpy.zeros(shape, dtype=numpy.float32)
        height, width = shape
        cell_height, cell_width = height // nrows, width // ncols
        for i in range(nrows):
            for j in range(ncols):
                i0 = cell_height * i + cell_height // 2 - radius
                j0 = cell_width * j + cell_width // 2 - radius
                arr[i0:i0+diameter,j0:j0+diameter] = circle
    else:
        import PIL.Image
        image = PIL.Image.open(mask_path)
        arr = numpy.array(image).astype(numpy.float32)
        image.close()
        if arr.ndim == 2:
            arr /= 255
        elif arr.ndim == 3:
            if arr.shape[2] == 4:
                warnings.warn(f"Mask {mask_path} has an alpha channel but it will be ignored")
            arr = numpy.mean(arr[:,:,:3], axis=2) / 255
        else:
            raise ValueError(f"Image has wrong number of dimensions {arr.ndim}, expected 2 or 3")
    if inverse:
        arr = 1.0 - arr
    return cast(FloatMask, arr)


def load_bool_mask(mask_path: str | None, shape: tuple[int, int] = (0, 0), default: bool = False) -> BoolMask:
    return cast(BoolMask, numpy.round(load_float_mask(mask_path, shape, float(default))).astype(numpy.bool))


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


# CSS colors, taken from matplotlib.colors.CSS4_COLORS
NAMED_COLORS = {
    "aliceblue": (240, 248, 255),
    "antiquewhite": (250, 235, 215),
    "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212),
    "azure": (240, 255, 255),
    "beige": (245, 245, 220),
    "bisque": (255, 228, 196),
    "black": (0, 0, 0),
    "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255),
    "blueviolet": (138, 43, 226),
    "brown": (165, 42, 42),
    "burlywood": (222, 184, 135),
    "cadetblue": (95, 158, 160),
    "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30),
    "coral": (255, 127, 80),
    "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220),
    "crimson": (220, 20, 60),
    "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139),
    "darkcyan": (0, 139, 139),
    "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkgrey": (169, 169, 169),
    "darkkhaki": (189, 183, 107),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0),
    "darkorchid": (153, 50, 204),
    "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79),
    "darkslategrey": (47, 79, 79),
    "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211),
    "deeppink": (255, 20, 147),
    "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34),
    "floralwhite": (255, 250, 240),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0),
    "goldenrod": (218, 165, 32),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (173, 255, 47),
    "grey": (128, 128, 128),
    "honeydew": (240, 255, 240),
    "hotpink": (255, 105, 180),
    "indianred": (205, 92, 92),
    "indigo": (75, 0, 130),
    "ivory": (255, 255, 240),
    "khaki": (240, 230, 140),
    "lavender": (230, 230, 250),
    "lavenderblush": (255, 240, 245),
    "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205),
    "lightblue": (173, 216, 230),
    "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255),
    "lightgoldenrodyellow": (250, 250, 210),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122),
    "lightseagreen": (32, 178, 170),
    "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153),
    "lightslategrey": (119, 136, 153),
    "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (250, 240, 230),
    "magenta": (255, 0, 255),
    "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170),
    "mediumblue": (0, 0, 205),
    "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219),
    "mediumseagreen": (60, 179, 113),
    "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154),
    "mediumturquoise": (72, 209, 204),
    "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112),
    "mintcream": (245, 255, 250),
    "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181),
    "navajowhite": (255, 222, 173),
    "navy": (0, 0, 128),
    "oldlace": (253, 245, 230),
    "olive": (128, 128, 0),
    "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0),
    "orangered": (255, 69, 0),
    "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170),
    "palegreen": (152, 251, 152),
    "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147),
    "papayawhip": (255, 239, 213),
    "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63),
    "pink": (255, 192, 203),
    "plum": (221, 160, 221),
    "powderblue": (176, 224, 230),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "rosybrown": (188, 143, 143),
    "royalblue": (65, 105, 225),
    "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114),
    "sandybrown": (244, 164, 96),
    "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238),
    "sienna": (160, 82, 45),
    "silver": (192, 192, 192),
    "skyblue": (135, 206, 235),
    "slateblue": (106, 90, 205),
    "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144),
    "snow": (255, 250, 250),
    "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180),
    "tan": (210, 180, 140),
    "teal": (0, 128, 128),
    "thistle": (216, 191, 216),
    "tomato": (255, 99, 71),
    "turquoise": (64, 224, 208),
    "violet": (238, 130, 238),
    "wheat": (245, 222, 179),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (255, 255, 0),
    "yellowgreen": (154, 205, 50)
}


def parse_color(string: str) -> tuple[int, int, int]:
    if string.lower() in NAMED_COLORS:
        return NAMED_COLORS[string.lower()]
    rgb_match = re.match(r"^(?:rgb)?\((\d+), ?(\d+), ?(\d+)\)$", string, re.IGNORECASE)
    if rgb_match is not None:
        return (int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3)))
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
