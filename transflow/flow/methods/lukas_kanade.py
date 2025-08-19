from typing import cast

import cv2
import numpy

from ...types import Grey, Flow


def calc_optical_flow_lukas_kanade(
        prev_grey: Grey,
        next_grey: Grey,
        win_size: int,
        max_level: int,
        step: int) -> Flow:
    m, n = prev_grey.shape
    p0 = numpy.stack(
        numpy.meshgrid(
            numpy.arange(0, prev_grey.shape[1], step),
            numpy.arange(0, prev_grey.shape[0], step),
            indexing="xy"),
        axis=-1)\
        .astype(numpy.float32)
    p, q = p0.shape[:2]
    p0 = p0.reshape((p * q, 1, 2))
    p1 = p0.copy()
    cv2.calcOpticalFlowPyrLK(
        prev_grey,
        next_grey,
        p0,
        p1,
        winSize=(win_size, win_size),
        maxLevel=max_level)
    flow = p1.reshape((p, q, 2)) - p0.reshape((p, q, 2))
    if step == 1:
        return cast(Flow, flow)
    return cast(Flow, numpy.kron(flow, numpy.ones((step, step, 1)))[0:m,0:n,:].astype(flow.dtype))
