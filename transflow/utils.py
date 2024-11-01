import os
import re

import numpy


def load_mask(mask_path: str, newaxis: bool = False) -> numpy.ndarray:
    import PIL.Image
    image = PIL.Image.open(mask_path)
    arr = numpy.array(image)
    image.close()
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
    if re.match(r".*\.(\d{3})$", root):
        i = int(re.match(r".*\.(\d{3})$", root).group(1)) + 1
        root = root[:-4]
    while os.path.isfile(path):
        path = root + f".{i:03d}" + ext
        i += 1
    return path


def parse_hex_color(string: str) -> tuple[int, int, int]:
    s = string.replace("#", "").replace("0x", "").replace("x", "")
    x = int(s, 16)
    return [(x >> 16) & 255, (x >> 8) & 255, x & 255]


def compose_top(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    return colors[-1]


def compose_additive(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    return [
        min(255, sum(c[0] for c in colors)),
        min(255, sum(c[1] for c in colors)),
        min(255, sum(c[2] for c in colors))
    ]


def compose_subtractive(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    c = [*colors[0]]
    for c2 in colors[1:]:
        for k in range(3):
            c[k] = max(0, c[k] - (255 - c2[k]))
    return c


def compose_average(*colors: tuple[int, int, int]) -> tuple[int, int, int]:
    n = len(colors)
    if n == 0:
        return (0, 0, 0)
    return (
        int(sum(c[0] for c in colors) / n),
        int(sum(c[1] for c in colors) / n),
        int(sum(c[2] for c in colors) / n),
    )
