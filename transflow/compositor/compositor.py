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


class Compositor:

    @enum.unique
    class ResetMode(enum.Enum):

        OFF = 0
        RANDOM = 1
        LINEAR = 2

        @classmethod
        def from_string(cls, string):
            match string:
                case "off":
                    return cls.OFF
                case "random":
                    return cls.RANDOM
                case "linear":
                    return cls.LINEAR
            raise ValueError(f"Unknown reset mode {string}")

    class Layer:

        def __init__(self, width: int, height: int):
            self.width = width
            self.height = height
            self.sources: list[BitmapSource] = []
            self.data = numpy.zeros((self.height, self.width, 8), dtype=numpy.int32)
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

        def _update_insert(self, flow: numpy.ndarray):
            raise NotImplementedError() # TODO

        def _update_move(self, flow: numpy.ndarray):
            """Quick recap of the process:
            1. Compute source and target masks
            2. Compute source and target indices of actually moving pixels
            3. If set, set the source of moving pixels to None
            4. Apply the movement
            """
            
            # TODO: those will probably be reused, so they could be stored as attributes
            flow_int = numpy.round(flow).astype(numpy.int32)
            flow_flat = numpy.ravel(flow_int[:,:,1] * self.width + flow_int[:,:,0])
            shift = numpy.arange(self.height * self.width) + flow_flat
                                    
            # 2D mask of places where pixels are allowed to move from
            # TODO: apply other masks on top
            mask_source = numpy.ones((self.height, self.width), dtype=numpy.bool)
            
            # Apply the flow to the mask so its indices are in the target space
            mask_source = mask_source.flat[shift].reshape((self.height, self.width))
            
            # 2D mask of places where pixels are allowed to move to
            # TODO: apply other masks on top
            mask_target = numpy.ones((self.height, self.width), dtype=numpy.bool)
            
            mask_all_flat = numpy.multiply(mask_source.flat, mask_target.flat)
            where_target = numpy.multiply(flow_flat, mask_all_flat)[0]
            where_source = where_target + flow_flat[where_target]
            
            crumble = True # TODO (and find another name!)
            if crumble:
                # Each pixel in where_source should be set to source None
                putn_1d(self.data, 0, where_source, 8, 0)
            
            putn(self.data, self.data.copy(), where_target, where_source, 8)
        
        def _update_reset(self, flow: numpy.ndarray):
            raise NotImplementedError() # TODO
        
        def _update_sources(self):
            """Update the RGBA values of each dynamic source.
            """
            for i, source in enumerate(self.sources):
                source.next()
                bitmap = source.get_bitmap()
                where = numpy.where(self.data[:,:,0] == i + 2)
                # self.data[:,:,4:4+bitmap.shape[2]] is the range of RGBA values in data attribute
                # (self.data[:,:,1], self.data[:,:,2]) are the indices of values used from bitmap in data attribute
                self.data[:,:,4:4+bitmap.shape[2]][where] = bitmap[self.data[:,:,1], self.data[:,:,2]][where]

        def update(self, flow: numpy.ndarray):
            self._update_insert(flow)
            self._update_move(flow)
            self._update_reset(flow)
            self._update_sources()

        def render(self) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]:
            """
            :return: RGBA array of shape (height, width, 4)
            """
            
            # Use RGBA data from attribute
            image = numpy.clip(self.data[:,:,4:8], 0, 255, dtype=numpy.uint8)
            
            # If source is None, force opacity to 0
            image[numpy.where(self.data[:,:,0]) == 0][3] = 0 
            
            return image

    def __init__(self, width: int, height: int, background_color: str = "#ffffff"):
        self.width = width
        self.height = height
        self.background_color = parse_hex_color(background_color)
        self.background = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        self.background[:,:] = self.background_color
        self.layers: list[Compositor.Layer] = [] # TODO: how are layers added/created?

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
