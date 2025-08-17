"""
https://stackoverflow.com/questions/9195455/how-to-document-a-method-with-parameters
"""
import enum
import logging
import multiprocessing
import numpy
import warnings
from collections.abc import Sequence

from ..utils import parse_hex_color
from ..config import LayerConfig


logger = logging.getLogger(__name__)


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
            width: int,
            height: int,
            sources: list[PixmapSourceInterface],
            ):
        self.width = width
        self.height = height
        self.sources: list[PixmapSourceInterface] = sources
        self.rgba = numpy.zeros((self.height, self.width, 4), dtype=numpy.uint8)

    def update(self, flow: numpy.ndarray):
        raise NotImplementedError()

    def render(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
        """
        :return: RGBA array of shape (height, width, 4)
        """
        # return numpy.clip(
        #     numpy.append(self.rgb, 255 * self.alpha.reshape(2, 3, 1), axis=2),
        #     0, 255, dtype=numpy.uint8)
        return numpy.clip(self.rgba, 0, 255, dtype=numpy.uint8)

    @classmethod
    def from_args(cls,
            config: LayerConfig,
            width: int,
            height: int,
            sources: list[PixmapSourceInterface]):
        args = [width, height, sources]
        kwargs = {}
        movement_kwargs = {
            "mask_src": config.mask_src,
            "mask_dst": config.mask_dst,
            "mask_alpha": config.mask_alpha,
            "transparent_pixels_can_move": config.transparent_pixels_can_move,
            "pixels_can_move_to_empty_spot": config.pixels_can_move_to_empty_spot,
            "pixels_can_move_to_filled_spot": config.pixels_can_move_to_filled_spot,
            "moving_pixels_leave_empty_spot": config.moving_pixels_leave_empty_spot,
        }
        reference_kwargs = {
            "reset_mask": config.reset_mask,
            "reset_mode": ResetMode.from_string(config.reset_mode),
            "reset_pixels_leave_healed_spot": config.reset_pixels_leave_healed_spot,
            "reset_pixels_leave_empty_spot": config.reset_pixels_leave_empty_spot,
            "reset_random_factor": config.reset_random_factor,
            "reset_constant_step": config.reset_constant_step,
            "reset_linear_factor": config.reset_linear_factor,
        }
        # NOTE: pass the whole LayerConfig instead of separated args?
        if not sources:
            warnings.warn("Layer has not sources!")
        if config.classname == "reference":
            return ReferenceLayer(*args, **reference_kwargs, **movement_kwargs, **kwargs)
        if config.classname == "introduction":
            return IntroductionLayer(*args, **movement_kwargs, **kwargs)
        raise ValueError(f"Unknown layer classname {config.classname}")


class MovementLayer(Layer):
    """A layer moving elements in a 2D array.
    """

    DEPTH: int = 4
    POS_I_IDX: int = 0
    POS_J_IDX: int = 1
    POS_A_IDX: int = 2

    def __init__(self,
            *args,
            mask_src: numpy.ndarray | None = None,
            mask_dst: numpy.ndarray | None = None,
            mask_alpha: numpy.ndarray | None = None,
            transparent_pixels_can_move: bool = False,
            pixels_can_move_to_empty_spot: bool = True,
            pixels_can_move_to_filled_spot: bool = True,
            moving_pixels_leave_empty_spot: bool = False,
            **kwargs):
        Layer.__init__(self, *args, **kwargs)

        self.mask_src: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if mask_src is None else mask_src
        self.mask_dst: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if mask_dst is None else mask_dst
        self.mask_alpha: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.bool) if mask_alpha is None else mask_alpha
        self.transparent_pixels_can_move: bool = transparent_pixels_can_move
        self.pixels_can_move_to_empty_spot: bool = pixels_can_move_to_empty_spot
        self.pixels_can_move_to_filled_spot: bool = pixels_can_move_to_filled_spot
        self.moving_pixels_leave_empty_spot: bool = moving_pixels_leave_empty_spot

        self.base = numpy.indices((self.height, self.width), dtype=numpy.int32).transpose(1, 2, 0)
        self.data = numpy.zeros((self.height, self.width, self.DEPTH), dtype=numpy.int32)
        self.initial_data = self.data.copy()
        # self.rgb = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        # self.alpha = self.mask_alpha.copy()
        self.flow = numpy.zeros((self.height, self.width, 2), dtype=numpy.float32)
        self.flow_int = numpy.zeros((self.height, self.width, 2), dtype=numpy.int32)
        self.flow_flat = numpy.zeros((self.height * self.width, 1), dtype=numpy.int32)

        ## self.data = numpy.zeros((self.height, self.width, 8), dtype=numpy.int32)
        ## Data Structure of the Third Dimension
        ## 0: source index (0 is None, 1 is static, 2+ are bitmap sources)
        ## 1: source I
        ## 2: source J
        ## 3: source t (for dynamic sources)
        ## 4: source R
        ## 5: source G
        ## 6: source B
        ## 7: source A
        ## TODO: initialize data

    def _post_init(self):
        self.initial_data = self.data.copy()

    def _update_flow(self, flow: numpy.ndarray):
        self.flow = flow
        self.flow_int = numpy.round(self.flow).astype(numpy.int32)
        self.flow_flat = numpy.ravel(self.flow_int[:,:,1] * self.width + self.flow_int[:,:,0])

    def _update_move(self):
        shift = numpy.arange(self.height * self.width) + self.flow_flat
        mask_src = self.mask_src.copy()

        if not self.transparent_pixels_can_move:
            mask_src[numpy.where(self.data[:,:,self.POS_A_IDX] == 0)] = 0
        mask_src = mask_src.flat[shift].reshape((self.height, self.width))

        mask_dst = self.mask_dst.copy()
        if not self.pixels_can_move_to_empty_spot:
            mask_dst[numpy.where(self.data[:,:,self.POS_A_IDX] == 0)] = 0
        if not self.pixels_can_move_to_filled_spot:
            mask_dst[numpy.nonzero(self.data[:,:,self.POS_A_IDX])] = 0

        mask_all_flat = numpy.multiply(mask_src.flat, mask_dst.flat)
        where_target = numpy.nonzero(numpy.multiply(self.flow_flat, mask_all_flat))[0]
        where_source = where_target + self.flow_flat[where_target]

        aux = self.data.copy()
        putn(self.data, aux, where_target, where_source, self.DEPTH)
        if self.moving_pixels_leave_empty_spot:
            putn_1d(self.data[:,:,self.POS_A_IDX], 0, where_source, 1, 0)
        putn_1d(self.data[:,:,self.POS_A_IDX], 1, where_target, 1, 0)

    # def _update_move(self):
    #     """Quick recap of the process:
    #     1. Compute source and target masks
    #     2. Compute source and target indices of actually moving pixels
    #     3. If set, set the source of moving pixels to None
    #     4. Apply the movement
    #     """
    #     # raise NotImplementedError()
    #     pass

    #     # # TODO: those will probably be reused, so they could be stored as attributes

    #     # shift = numpy.arange(self.height * self.width) + self.flow_flat

    #     # # 2D mask of places where pixels are allowed to move from
    #     # # TODO: apply other masks on top
    #     # mask_source = numpy.ones((self.height, self.width), dtype=numpy.bool)

    #     # # Apply the flow to the mask so its indices are in the target space
    #     # mask_source = mask_source.flat[shift].reshape((self.height, self.width))

    #     # # 2D mask of places where pixels are allowed to move to
    #     # # TODO: apply other masks on top
    #     # mask_target = numpy.ones((self.height, self.width), dtype=numpy.bool)

    #     # mask_all_flat = numpy.multiply(mask_source.flat, mask_target.flat)
    #     # where_target = numpy.multiply(self.flow_flat, mask_all_flat)[0]
    #     # where_source = where_target + self.flow_flat[where_target]

    #     # crumble = True # TODO (and find another name!)
    #     # if crumble:
    #     #     # Each pixel in where_source should be set to source None
    #     #     putn_1d(self.rgb, 0, where_source, 8, 0)

    #     # putn(self.rgb, self.rgb.copy(), where_target, where_source, 8)

    # def _update_reset_random(self):
    #     # threshold = numpy.ones((self.height, self.width), dtype=numpy.float32) # TODO
    #     # random = numpy.random.random(size=(self.height, self.width))
    #     # where = numpy.where(random < threshold)
    #     # aux = self.rgb.copy()
    #     # crumble = True # TODO
    #     # if crumble:
    #     #     self.rgb[*where,0] = 0
    #     # self.rgb[aux[:,:,1], aux[:,:,2]][where] = aux[where]
    #     pass

    # def _apply_reset_movement(self, movement_flat: numpy.ndarray, reset_mask: numpy.ndarray):
        
    #     where_target = numpy.nonzero(numpy.multiply(movement_flat, reset_mask.flat))[0]
    #     where_source = numpy.clip(where_target + movement_flat[where_target], 0, self.height * self.width - 1)
        
        
    #     # it is not a target, ie. nothing  ## BUT: target should be everywhere!!!
    #     # it is not a source, ie. nothing refers to it anymore
    
    #     # where_source_not_target = numpy.setdiff1d(where_source, where_target)
    #     # # locations that are both
    #     # where_both_target = numpy.intersect1d(where_target, where_source)
    #     # where_both_source = where_both_target + movement_flat[where_both_target]
    #     # print("\n", where_both_target.shape[0])

    #     # aux = self.data.copy()
    #     aux = self.data.copy()
    #     if self.reset_pixels_leave_healed_spot:
    #         putn(self.data, self.initial_data, where_source, where_source, self.DEPTH)
    #     elif self.reset_pixels_leave_empty_spot:
    #         putn_1d(self.data[:,:,self.POS_A_IDX], 0, where_source, 1, 0)
    #     putn_1d(self.data[:,:,self.POS_A_IDX], 1, where_target, 1, 0)
    #     putn(self.data, aux, where_target, where_source, self.DEPTH)
        
    #     # if self.reset_pixels_leave_healed_spot:
    #     #     putn(self.data, self.initial_data, where_source, where_source, self.DEPTH)
    #     # elif self.reset_pixels_leave_empty_spot:
    #     #     putn_1d(self.data[:,:,self.POS_A_IDX], 0, where_source, 1, 0)
                
    #     # TODO: fix reset movement!!!

    

    def _update_rgba(self):
        raise NotImplementedError()

    def update(self, flow: numpy.ndarray):
        self._update_flow(flow)
        self._update_move()
        self._update_rgba()


class ReferenceLayer(MovementLayer):

    DEPTH = 4 # i, j, alpha, source
    POS_I_IDX: int = 0
    POS_J_IDX: int = 1
    POS_A_IDX: int = 2

    def __init__(self,
            *args,
            reset_mode: ResetMode = ResetMode.OFF,
            reset_mask: numpy.ndarray | None = None,
            reset_pixels_leave_healed_spot: bool = True,
            reset_pixels_leave_empty_spot: bool = True,
            reset_random_factor: float = 1,
            reset_constant_step: float = 1,
            reset_linear_factor: float = 0.1,
            **kwargs):
        MovementLayer.__init__(self, *args, **kwargs)
        self.data[:,:,0:2] = self.base.copy() # shape: (height, width, 2) [i, j]
        self.data[:,:,2] = 1
        self.data[:,:,3] = 0 # source index
        self._post_init()
        
        self.reset_mode = reset_mode
        self.reset_mask: numpy.ndarray = numpy.ones((self.height, self.width), dtype=numpy.float32) if reset_mask is None else reset_mask
        self.reset_pixels_leave_healed_spot: bool = reset_pixels_leave_healed_spot
        self.reset_pixels_leave_empty_spot: bool = reset_pixels_leave_empty_spot
        self.reset_random_factor: float = reset_random_factor
        self.reset_constant_step: float = reset_constant_step # value in pixels
        self.reset_linear_factor: float = reset_linear_factor # 0: no movement / 1: instant reset
        
        # NOTE
        # summation is completely different. it does not take alpha into account.
        # we could have a third layer class 'SumLayer', but would masks be correctly taken into account?
        # it should inherit from the same base class as MappingLayer
        
    def _update_reset_random(self):

        # NOTE
        # # This is the very basic behavior: some pixels get set back to their original value
        # where = numpy.where(numpy.random.random(size=(self.height, self.width)) < self.reset_mask)
        # self.mapping[where] = self.base[where]
        # self.mask_alpha[where] = 1 # TODO: refer to original mask_alpha?

        # NOTE
        # Another potential behavior: some pixels GO BACK to their original position
        # ie. move a selection of pixels to their original location
        # this allows for 'self.moving_pixels_leave_empty_spot'
        # and is more general.
        # the thing is the orignal mask differs,
        # as the original mask is more of source mask
        # but it can be inverted? # TODO

        # NOTE
        # the mecanism described above seems like the right thing to do:
        # moving pixels back to their original position.
        # yet, the process is destructive: some pixels get overwritten.
        # more than that, we can have a situation where the pixel is has to go back to has'nt changed
        # but the spot it leaves when going back has just disappared.
        # in that case, we have multiple options:
        #  (i)   do nothing (leave the copy of the pixel / that's half a reset IMO)
        #  (ii)  leave an empty spot
        #  (iii) reset the mapping to the base value

        # NOTE
        # in non-random reset (ie. constant or linear reset), the reset is once
        # again considered as a movement toward the original position.
        # the same logic should apply?
        # one issue I can think of, in case option (iii) 'RESET TO BASE VALUE'
        # is not selected, is that resetting pixels to their positions will
        # AGAIN proceed to delete them.
        # that's probably worth trying.

        random = numpy.random.random(size=(self.height, self.width))
        reset_mask = numpy.zeros((self.height, self.width), dtype=numpy.bool)
        reset_mask[numpy.where(random < self.reset_random_factor * self.reset_mask)] = 1
        # movement_flat = numpy.arange(self.width * self.height) - numpy.ravel(self.data[:,:,self.POS_I_IDX] * self.width + self.data[:,:,self.POS_J_IDX])
        
        where = numpy.nonzero(reset_mask)
        self.data[where] = self.initial_data[where]
        
        
        # # TODO: remove. This is for testing. Fake flow, movement to the left
        # x = numpy.ones((self.height, self.width), dtype=numpy.int32)
        # x[:,-1] = 0
        # movement_flat = numpy.ravel(x)
        
        # self._apply_reset_movement(movement_flat, reset_mask)

    def _update_reset_constant(self):
        dij = self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)]
        dij_norm = numpy.linalg.norm(dij, ord=float("inf"), axis=2)
        dij_norm[numpy.where(dij_norm > self.reset_constant_step)] /= self.reset_constant_step # make sure to not overshoot
        where = numpy.nonzero(dij_norm)
        dij_scaled = dij.copy()        
        dij_scaled[where] = dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]
        # dij_scaled[where] = self.reset_constant_step * dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij_scaled
        # dij_scaled = dij.copy()
        # where = numpy.nonzero(dij_norm)
        # dij_scaled[where] = self.reset_constant_step * dij[where] / dij_norm.reshape((self.height, self.width, 1))[where]
        # movement_flat = numpy.ravel(dij_scaled[:,:,0] * self.width + dij_scaled[:,:,1])
        # self._apply_reset_movement(movement_flat, self.reset_mask.copy())

    def _update_reset_linear(self):
        dij = numpy.round(self.reset_linear_factor * (self.base - self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)])).astype(numpy.int32)
        self.data[:,:,[self.POS_I_IDX, self.POS_J_IDX]] += dij
        # dij = self.reset_linear_factor * (self.data[:,:,(self.POS_I_IDX, self.POS_J_IDX)] - self.base)
        # movement_flat = numpy.ravel(dij[:,:,0] * self.width + dij[:,:,1])
        # self._apply_reset_movement(movement_flat, self.reset_mask.copy())

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
            
    def update(self, flow: numpy.ndarray):
        self._update_flow(flow)
        self._update_move()
        self._update_reset()
        self._update_rgba()


class IntroductionLayer(MovementLayer):

    DEPTH = 8 # r, g, b, alpha, source, i, j, frame
    POS_I_IDX: int = 5
    POS_J_IDX: int = 6
    POS_A_IDX: int = 3

    def __init__(self, *args, **kwargs):
        MovementLayer.__init__(self, *args, **kwargs)
        self._post_init()
        self.introduction_masks: list[numpy.ndarray] = []
        for _ in self.sources:
            self.introduction_masks.append(numpy.ones((self.height, self.width), dtype=numpy.bool))
        self.introduce_pixels_on_empty_spots: bool = True
        self.introduce_pixels_on_filled_spots: bool = True
        self.introduce_moving_pixels: bool = True
        self.introduce_unmoving_pixels: bool = True

    def _update_introduction(self):
        mask = numpy.ones((self.height, self.width), dtype=numpy.bool) #self.mask_introduction.copy()
        if not self.introduce_pixels_on_empty_spots:
            mask[numpy.where(self.data[:,:,3]) == 0] = 0
        if not self.introduce_pixels_on_filled_spots:
            mask[numpy.nonzero(self.data[:,:,3])] = 0
        if not self.introduce_moving_pixels:
            mask.flat[numpy.nonzero(self.flow_flat)] = 0
        if not self.introduce_unmoving_pixels:
            mask.flat[numpy.where(self.flow_flat) == 0] = 0
        for i, (source, mask_introduction) in enumerate(zip(self.sources, self.introduction_masks)):
            pixmap = source.next()
            where_target = numpy.nonzero(numpy.multiply(mask.flat, mask_introduction.flat))[0]
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

    def update(self, flow: numpy.ndarray):
        self._update_flow(flow)
        self._update_move()
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
            image[where_opaque] = layer_image[:,:,:3][where_opaque]
        return image

    @classmethod
    def from_args(cls,
            width: int,
            height: int,
            layer_configs: list[LayerConfig],
            pixmap_interfaces: dict[int, list[PixmapSourceInterface]]):
        layers = [
            Layer.from_args(config, width, height, pixmap_interfaces.get(config.index, []))
            for config in layer_configs
        ]
        return cls(width, height, layers)
