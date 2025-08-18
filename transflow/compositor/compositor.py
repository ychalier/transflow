"""
https://stackoverflow.com/questions/9195455/how-to-document-a-method-with-parameters
"""
import enum
import logging
import multiprocessing
import warnings
from collections.abc import Sequence
from typing import Literal, TypeVar, cast

import numpy

from ..utils import parse_hex_color, load_mask
from ..config import LayerConfig


logger = logging.getLogger(__name__)

Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
Rgba = numpy.ndarray[tuple[Height, Width, Literal[4]], numpy.dtype[numpy.uint8]]
Flow = numpy.ndarray[tuple[Height, Width, Literal[2]], numpy.dtype[numpy.float32]]


class EndOfPixmap(StopIteration):
    pass


class PixmapSourceInterface:

    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue
        self.image: numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]] | None = None
        self.counter: int = -1

    def get(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        assert self.image is not None
        return self.image

    def next(self, timeout: float = 1) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        image = self.queue.get(timeout=timeout)
        if image is None:
            raise EndOfPixmap
        assert isinstance(image, numpy.ndarray)
        assert len(image.shape) == 3
        assert image.dtype == numpy.uint8
        self.image = image
        self.counter += 1
        return self.image

    @property
    def frame_number(self) -> int:
        return self.counter


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
    """Base class for a layer.
    """

    def __init__(self,
            config: LayerConfig,
            width: int,
            height: int,
            sources: list[PixmapSourceInterface],
            ):
        self.config = config
        self.width = width
        self.height = height
        self.sources: list[PixmapSourceInterface] = sources
        self.rgba = numpy.zeros((self.height, self.width, 4), dtype=numpy.uint8)
        self.mask_alpha: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if self.config.mask_alpha is None else load_mask(self.config.mask_alpha)

    def update(self, flow: Flow):
        raise NotImplementedError()

    def render(self) -> Rgba:
        self.rgba[:,:,3] = 255 * self.mask_alpha
        return cast(Rgba, numpy.clip(self.rgba, 0, 255).astype(numpy.uint8))

    @classmethod
    def from_args(cls,
            config: LayerConfig,
            width: int,
            height: int,
            sources: list[PixmapSourceInterface]):
        args = [config, width, height, sources]
        if config.classname == "moveref":
            return MoveReferenceLayer(*args)
        if config.classname == "introduction":
            return IntroductionLayer(*args)
        if config.classname == "static":
            return StaticLayer(*args)
        if config.classname == "sum":
            return SumLayer(*args)
        raise ValueError(f"Unknown layer classname {config.classname}")


class StaticLayer(Layer):

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.rgba[:,:,3] = 1

    def update(self, flow: Flow):
        for source in self.sources:
            pixmap = source.next()
            self.rgba[:,:,:pixmap.shape[2]] = pixmap


class DataLayer(Layer):

    DEPTH: int = 4
    POS_I_IDX: int = 0
    POS_J_IDX: int = 1
    POS_A_IDX: int = 2

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.base = numpy.indices((self.height, self.width), dtype=numpy.int32).transpose(1, 2, 0)
        self.data = numpy.zeros((self.height, self.width, self.DEPTH), dtype=numpy.int32)


class MovementLayer(DataLayer):

    def __init__(self, *args):
        Layer.__init__(self, *args)
        self.mask_src: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if self.config.mask_src is None else load_mask(self.config.mask_src)
        self.mask_dst: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if self.config.mask_dst is None else load_mask(self.config.mask_dst)
        self.flow = numpy.zeros((self.height, self.width, 2), dtype=numpy.float32)
        self.flow_int = numpy.zeros((self.height, self.width, 2), dtype=numpy.int32)
        self.flow_flat = numpy.zeros((self.height * self.width, 1), dtype=numpy.int32)

    def _update_flow(self, flow: Flow):
        self.flow = flow
        self.flow_int = numpy.round(self.flow).astype(numpy.int32)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])

    def _update_move(self):
        shift = numpy.arange(self.height * self.width) + self.flow_flat
        mask_src = self.mask_src.copy()

        if not self.config.transparent_pixels_can_move:
            mask_src[numpy.where(self.data[:,:,self.POS_A_IDX] == 0)] = 0
        mask_src = mask_src.flat[shift].reshape((self.height, self.width))

        mask_dst = self.mask_dst.copy()
        if not self.config.pixels_can_move_to_empty_spot:
            mask_dst[numpy.where(self.data[:,:,self.POS_A_IDX] == 0)] = 0
        if not self.config.pixels_can_move_to_filled_spot:
            mask_dst[numpy.nonzero(self.data[:,:,self.POS_A_IDX])] = 0

        mask_all_flat = numpy.multiply(mask_src.flat, mask_dst.flat)
        where_target = numpy.nonzero(numpy.multiply(self.flow_flat, mask_all_flat))[0]
        where_source = where_target + self.flow_flat[where_target]

        aux = self.data.copy()
        putn(self.data, aux, where_target, where_source, self.DEPTH)
        if self.config.moving_pixels_leave_empty_spot:
            putn_1d(self.data[:,:,self.POS_A_IDX], 0, where_source, 1, 0)
        putn_1d(self.data[:,:,self.POS_A_IDX], 1, where_target, 1, 0)

    def update(self, flow: Flow):
        self._update_flow(flow)
        self._update_move()


class ReferenceLayer(DataLayer):

    def __init__(self, *args):
        DataLayer.__init__(self, *args)
        self.data[:,:,0:2] = self.base.copy() # shape: (height, width, 2) [i, j]
        self.data[:,:,2] = 1
        self.data[:,:,3] = 0 # source index
        # NOTE
        # summation is completely different. it does not take alpha into account.
        # we could have a third layer class 'SumLayer', but would masks be correctly taken into account?
        # it should inherit from the same base class as MappingLayer
        self.reset_mode: ResetMode = ResetMode.from_string(self.config.reset_mode)
        self.reset_mask: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.float32) if self.config.reset_mask is None else load_mask(self.config.reset_mask)

    def _update_reset_random(self):
        random = numpy.random.random(size=(self.height, self.width))
        reset_mask = numpy.zeros((self.height, self.width), dtype=numpy.bool)
        reset_mask[numpy.where(random < self.config.reset_random_factor * self.reset_mask)] = 1
        where = numpy.nonzero(reset_mask)
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]][where] = self.base[where]
        self.data[:,:,self.POS_A_IDX][where] = 1

    def _update_reset_constant(self):
        dij = self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)]
        dij_norm = numpy.linalg.norm(dij, ord=float("inf"), axis=2)
        dij_norm[numpy.where(dij_norm > self.config.reset_constant_step)] /= self.config.reset_constant_step
        where = numpy.nonzero(dij_norm)
        dij_scaled = dij.copy()
        dij_scaled[where] = dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij_scaled

    def _update_reset_linear(self):
        dij = numpy.round(self.config.reset_linear_factor * (self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)])).astype(numpy.int32)
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij

    def _update_reset(self):
        if self.reset_mode == ResetMode.RANDOM:
            self._update_reset_random()
        elif self.reset_mode == ResetMode.CONSTANT:
            self._update_reset_constant()
        elif self.reset_mode == ResetMode.LINEAR:
            self._update_reset_linear()

    def _update_rgba(self):
        for i, source in enumerate(self.sources):
            where = numpy.where(self.data[:,:,3] == i)
            pixmap = source.next()
            mapping_i = numpy.clip(numpy.round(self.data[:,:,0]), 0, self.height - 1)[where]
            mapping_j = numpy.clip(numpy.round(self.data[:,:,1]), 0, self.width - 1)[where]
            self.rgba[:,:,:pixmap.shape[2]][where] = pixmap[mapping_i, mapping_j]
            self.rgba[:,:,3] = self.data[:,:,2] # TODO: pixmap alpha channel (if any) gets overwritten

    def update(self, flow: Flow):
        self._update_reset()
        self._update_rgba()


class MoveReferenceLayer(MovementLayer, ReferenceLayer):

    def __init__(self, *args):
        MovementLayer.__init__(self, *args)
        ReferenceLayer.__init__(self, *args)

    def update(self, flow: Flow):
        MovementLayer.update(self, flow)
        ReferenceLayer.update(self, flow)


class SumLayer(ReferenceLayer):

    def _update_sum(self, flow: Flow):
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += numpy.floor(flow).astype(numpy.int32)

    def update(self, flow: Flow):
        self._update_sum(flow)
        ReferenceLayer.update(self, flow)


class IntroductionLayer(MovementLayer):

    DEPTH = 8 # r, g, b, alpha, source, i, j, frame
    POS_I_IDX: int = 5
    POS_J_IDX: int = 6
    POS_A_IDX: int = 3

    def __init__(self, *args):
        MovementLayer.__init__(self, *args)
        # TODO: consider using one mask per source
        self.mask_introduction: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if self.config.mask_introduction is None else load_mask(self.config.mask_introduction)
        # self.introduction_masks: list[numpy.ndarray] = []
        # for _ in self.sources:
        #     self.introduction_masks.append(numpy.ones((self.height, self.width), dtype=numpy.bool))
        self.introduced_once: bool = False

    def _update_introduction(self):
        if self.config.introduce_once and self.introduced_once:
            return
        self.introduced_once = True
        mask = numpy.ones((self.height, self.width), dtype=numpy.bool) #self.mask_introduction.copy()
        if not self.config.introduce_pixels_on_empty_spots:
            mask[numpy.where(self.data[:,:,3]) == 0] = 0
        if not self.config.introduce_pixels_on_filled_spots:
            mask[numpy.nonzero(self.data[:,:,3])] = 0
        if not self.config.introduce_moving_pixels:
            mask.flat[numpy.nonzero(self.flow_flat)] = 0
        if not self.config.introduce_unmoving_pixels:
            mask.flat[numpy.where(self.flow_flat) == 0] = 0
        for i, source in enumerate(self.sources):
            pixmap = source.next()
            where_target = numpy.nonzero(numpy.multiply(mask.flat, self.mask_introduction.flat))[0]
            where_source = where_target + self.flow_flat[where_target]
            arrays = [
                pixmap,
                numpy.broadcast_to(i, (self.height, self.width, 1)),
                self.base,
                numpy.broadcast_to(source.frame_number, (self.height, self.width, 1)),
            ]
            if pixmap.shape[2] == 3:
                arrays.insert(1, numpy.broadcast_to(1, (self.height, self.width, 1)))
            putn(self.data, numpy.concat(arrays, axis=2), where_target, where_source, 8)

    def _update_rgba(self):
        self.rgba = self.data[:,:,:4]

    def update(self, flow: Flow):
        MovementLayer.update(self, flow)
        self._update_introduction()
        self._update_rgba()


class Compositor:

    def __init__(self, width: int, height: int, layers: Sequence[Layer], background_color: str = "#ffffff"):
        self.width = width
        self.height = height
        self.background_color = parse_hex_color(background_color)
        self.background = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        self.background[:,:] = self.background_color
        self.layers: Sequence[Layer] = layers

    def update(self, flow: Flow):
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
            image[where_opaque] = layer_image[:,:,:3][where_opaque]
        return image

    @classmethod
    def from_args(cls,
            width: int,
            height: int,
            layer_configs: list[LayerConfig]):
        layers = [Layer.from_args(config, width, height, []) for config in layer_configs]
        return cls(width, height, layers)

    def set_sources(self, pixmap_interfaces: dict[int, list[PixmapSourceInterface]]):
        for i, layer in enumerate(self.layers):
            layer.sources = pixmap_interfaces.get(i, [])
