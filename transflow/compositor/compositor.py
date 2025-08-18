import logging
from collections.abc import Sequence

import numpy

from ..utils import parse_hex_color
from ..types import Flow
from ..config import LayerConfig
from .layers import Layer
from .pixmap_source_interface import PixmapSourceInterface


logger = logging.getLogger(__name__)


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
