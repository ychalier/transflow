import cv2
import numpy


def calc_optical_flow_horn_schunck(
        prev_grey: numpy.ndarray,
        next_grey: numpy.ndarray,
        flow: numpy.ndarray | None = None,
        alpha: float = 1,
        max_iters: int = 3,
        decay: float = 0,
        delta: float = 1):
    import scipy.ndimage
    a = cv2.GaussianBlur(prev_grey.astype(numpy.float32), (5, 5), 0)
    b = cv2.GaussianBlur(next_grey.astype(numpy.float32), (5, 5), 0)
    if flow is None:
        u = numpy.zeros(a.shape)
        v = numpy.zeros(a.shape)
    else:
        u = decay * flow[:,:,0]
        v = decay * flow[:,:,1]
    x_kernel = numpy.array([[1, -1], [1, -1]]) * 0.25
    y_kernel = numpy.array([[1, 1], [-1, -1]]) * 0.25
    t_kernel = numpy.ones((2, 2)) * 0.25
    avg_kernel = numpy.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12
    ex = scipy.ndimage.convolve(a, x_kernel) + scipy.ndimage.convolve(b, x_kernel)
    ey = scipy.ndimage.convolve(a, y_kernel) + scipy.ndimage.convolve(b, y_kernel)
    et = scipy.ndimage.convolve(b, t_kernel) - scipy.ndimage.convolve(a, t_kernel)
    for _ in range(max_iters):
        u_avg = scipy.ndimage.convolve(u, avg_kernel)
        v_avg = scipy.ndimage.convolve(v, avg_kernel)
        c = numpy.divide(
            numpy.multiply(ex, u_avg) + numpy.multiply(ey, v_avg) + et,
            alpha ** 2 + numpy.pow(ex, 2) + numpy.pow(ey, 2)
        )
        prev = u
        u = u_avg - numpy.multiply(ex, c)
        v = v_avg - numpy.multiply(ey, c)
        if delta is not None and numpy.linalg.norm(u - prev, 2) < delta:
            break
    return numpy.stack([u, v], axis=-1).astype(numpy.float32)
