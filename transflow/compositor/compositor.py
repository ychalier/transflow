"""
https://stackoverflow.com/questions/9195455/how-to-document-a-method-with-parameters
"""
import enum
import logging

import numpy

from ..utils import parse_hex_color


logger = logging.getLogger(__name__)


class BitmapSource:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def get_bitmap(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        arr = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        # TODO
        return arr

    def next(self):
        pass # TODO


def putn(target_array: numpy.ndarray, source_array: numpy.ndarray, target_inds: numpy.ndarray, source_inds: numpy.ndarray, scale: int):
    target_inds_scaled = target_inds * scale
    source_inds_scaled = source_inds * scale
    for i in range(scale):
        target_array.flat[target_inds_scaled + i] = source_array.flat[source_inds_scaled + i]


def putn_1d(target_array: numpy.ndarray, value: int | float, target_inds: numpy.ndarray, scale: int, axis: int):
    target_inds_scaled = target_inds * scale
    target_array.flat[target_inds_scaled + axis] = value


@enum.unique
class ResetMode(enum.Enum):

    OFF = 0
    RANDOM = 1
    CONSTANT = 2
    LINEAR = 3

    @classmethod
    def from_string(cls, string):
        match string:
            case "off":
                return cls.OFF
            case "random":
                return cls.RANDOM
            case "constant":
                return cls.CONSTANT
            case "linear":
                return cls.LINEAR
        raise ValueError(f"Unknown reset mode {string}")


class Layer:

    def __init__(self, width: int, height: int, reset_mode: ResetMode):
        self.width = width
        self.height = height
        self.reset_mode = reset_mode
        self.sources: list[BitmapSource] = []
        self.base = numpy.indices((self.height, self.width), dtype=numpy.int32).transpose(1, 2, 0)
        self.rgb = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)

        # A bitmap of where pixels can come from
        self.mask_src = numpy.ones((self.height, self.width), dtype=numpy.bool)

        # A bitmap of where pixels can go to
        self.mask_dst = numpy.ones((self.height, self.width), dtype=numpy.bool)

        # The base alpha channel
        self.mask_alpha = numpy.ones((self.height, self.width), dtype=numpy.bool)

        self.flow = numpy.zeros((self.height, self.width, 2), dtype=numpy.float32)
        self.flow_int = numpy.zeros((self.height, self.width, 2), dtype=numpy.int32)
        self.flow_flat = numpy.zeros((self.height * self.width, 1), dtype=numpy.int32)



        # self.data = numpy.zeros((self.height, self.width, 8), dtype=numpy.int32)
        # Data Structure of the Third Dimension
        # 0: source index (0 is None, 1 is static, 2+ are bitmap sources)
        # 1: source I
        # 2: source J
        # 3: source t (for dynamic sources)
        # 4: source R
        # 5: source G
        # 6: source B
        # 7: source A
        # TODO: initialize data

    def _update_flow(self, flow: numpy.ndarray):
        self.flow = flow
        self.flow_int = numpy.round(self.flow).astype(numpy.int32)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])

    def _update_insert(self):
        raise NotImplementedError() # TODO

    def _update_move(self):
        """Quick recap of the process:
        1. Compute source and target masks
        2. Compute source and target indices of actually moving pixels
        3. If set, set the source of moving pixels to None
        4. Apply the movement
        """

        # TODO: those will probably be reused, so they could be stored as attributes

        shift = numpy.arange(self.height * self.width) + self.flow_flat

        # 2D mask of places where pixels are allowed to move from
        # TODO: apply other masks on top
        mask_source = numpy.ones((self.height, self.width), dtype=numpy.bool)

        # Apply the flow to the mask so its indices are in the target space
        mask_source = mask_source.flat[shift].reshape((self.height, self.width))

        # 2D mask of places where pixels are allowed to move to
        # TODO: apply other masks on top
        mask_target = numpy.ones((self.height, self.width), dtype=numpy.bool)

        mask_all_flat = numpy.multiply(mask_source.flat, mask_target.flat)
        where_target = numpy.multiply(self.flow_flat, mask_all_flat)[0]
        where_source = where_target + self.flow_flat[where_target]

        crumble = True # TODO (and find another name!)
        if crumble:
            # Each pixel in where_source should be set to source None
            putn_1d(self.rgb, 0, where_source, 8, 0)

        putn(self.rgb, self.rgb.copy(), where_target, where_source, 8)

    def _update_reset_random(self):
        threshold = numpy.ones((self.height, self.width), dtype=numpy.float32) # TODO
        random = numpy.random.random(size=(self.height, self.width))
        where = numpy.where(random < threshold)
        aux = self.rgb.copy()
        crumble = True # TODO
        if crumble:
            self.rgb[*where,0] = 0
        self.rgb[aux[:,:,1], aux[:,:,2]][where] = aux[where]

    def _update_reset_constant(self):
        # Where the source is not None
        # TODO
        # where = numpy.nonzero(self.data[:,:,0])

        # dij[i, j] = (di, dj) is the move to apply to that pixel for it to return to its position
        dij = self.rgb[:,:,1:3] - self.base

        # TODO: parameter to control reset speed
        speed = 1

        dij_norm = numpy.linalg.norm(dij, axis=2)
        dij_scaled = dij.copy()
        where = numpy.nonzero(dij_norm)
        dij_scaled[where] = speed * dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]

        aux = self.rgb.copy()

        crumble = True # TODO
        if crumble:
            pass # TODO

        # dij_scaled[i, j] + self.base is where to put each pixel
        # TODO: test this in Numpy?
        self.rgb[dij_scaled + self.base] = aux

        pass # TODO

    def _update_reset_linear(self):
        pass # TODO

    def _update_reset(self):
        if self.reset_mode == ResetMode.RANDOM:
            self._update_reset_random()
        elif self.reset_mode == ResetMode.CONSTANT:
            self._update_reset_constant()
        elif self.reset_mode == ResetMode.LINEAR:
            self._update_reset_linear()

    def _update_sources(self):
        """Update the RGBA values of each dynamic source.
        """
        for i, source in enumerate(self.sources):
            source.next()
            bitmap = source.get_bitmap()
            where = numpy.where(self.rgb[:,:,0] == i + 2)
            # self.data[:,:,4:4+bitmap.shape[2]] is the range of RGBA values in data attribute
            # (self.data[:,:,1], self.data[:,:,2]) are the indices of values used from bitmap in data attribute
            self.rgb[:,:,4:4+bitmap.shape[2]][where] = bitmap[self.rgb[:,:,1], self.rgb[:,:,2]][where]

    def update(self, flow: numpy.ndarray):
        self._update_flow(flow)
        # self._update_insert() # TODO
        self._update_move()
        self._update_reset()
        self._update_sources()

    def render(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        """
        :return: RGBA array of shape (height, width, 4)
        """
        return numpy.clip(
            numpy.append(self.rgb, 255 * self.mask_alpha.reshape(2, 3, 1), axis=2),
            0, 255, dtype=numpy.uint8)


class MappingLayer(Layer):

    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, **kwargs)
        assert len(self.sources) == 1
        self.source = self.sources[0]
        self.mapping = self.base.copy() # shape: (height, width, 2) [i, j]
        self.transparent_pixels_can_move: bool = False
        self.pixels_can_move_to_empty_spot: bool = True
        self.pixels_can_move_to_filled_spot: bool = True
        self.moving_pixels_leave_empty_spot: bool = False

    def _update_move(self):

        shift = numpy.arange(self.height * self.width) + self.flow_flat
        mask_src = self.mask_src.copy()

        if not self.transparent_pixels_can_move:
            mask_src[numpy.where(self.mask_alpha == 0)] = 0
        mask_src = mask_src.flat[shift].reshape((self.height, self.width))

        mask_dst = self.mask_dst.copy()
        if not self.pixels_can_move_to_empty_spot:
            mask_dst[numpy.where(self.mask_alpha == 0)] = 0
        if not self.pixels_can_move_to_filled_spot:
            mask_dst[numpy.nonzero(self.mask_alpha)] = 0
        
        mask_all_flat = numpy.multiply(mask_src.flat, mask_dst.flat)
        where_target = numpy.nonzero(numpy.multiply(self.flow_flat, mask_all_flat))[0]
        where_source = where_target + self.flow_flat[where_target]
        
        if self.moving_pixels_leave_empty_spot:
            putn_1d(self.mask_alpha, 0, where_source, 1, 0)
            putn_1d(self.mask_alpha, 1, where_target, 1, 0)
        putn(self.rgb, self.rgb.copy(), where_target, where_source, 3)

    def _update_sources(self):
        self.source.next()
        pixmap = self.source.get_bitmap()
        mapping_i = numpy.clip(numpy.round(self.mapping[:,:,0]), 0, self.height - 1)
        mapping_j = numpy.clip(numpy.round(self.mapping[:,:,1]), 0, self.width - 1)
        self.rgb[:,:,pixmap.shape[2]] = pixmap[mapping_i, mapping_j]


class CanvasLayer(Layer):
    pass


class Compositor:

    def __init__(self, width: int, height: int, background_color: str = "#ffffff"):
        self.width = width
        self.height = height
        self.background_color = parse_hex_color(background_color)
        self.background = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        self.background[:,:] = self.background_color
        self.layers: list[Layer] = [] # TODO: how are layers added/created?

    def update(self, flow: numpy.ndarray):
        for layer in self.layers:
            layer.update(flow)

    def render(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        """
        :return: RGB array of shape (height, width, 3)
        """
        image = self.background.copy()
        for layer in self.layers:
            layer_image = layer.render()
            where_opaque = numpy.nonzero(layer_image[:,:,3])
            image[where_opaque] = layer_image[where_opaque][:,:,:3]
        return image
