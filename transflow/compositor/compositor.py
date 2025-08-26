import logging
from collections.abc import Sequence
from typing import cast

import numpy

from ..utils import parse_color
from ..types import Flow, Rgb
from ..config import LayerConfig
from .layers import Layer
from .pixmap_source_interface import PixmapSourceInterface


logger = logging.getLogger(__name__)


class Compositor:

    def __init__(self, height: int, width: int, layers: Sequence[Layer], background_color: str = "#ffffff"):
        self.height = height
        self.width = width
        self.background_color = parse_color(background_color)
        self.background: Rgb = cast(Rgb, numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8))
        self.background[:,:] = self.background_color
        self.layers: Sequence[Layer] = layers

    def update(self, flow: Flow):
        for layer in self.layers:
            layer.update(flow)

    def render(self) -> Rgb:
        """
        :return: RGB array of shape (height, width, 3)
        """
        image = self.background.copy()
        for layer in self.layers:
            layer_image = layer.render()
            where_opaque = numpy.nonzero(layer_image[:,:,3])
            image[where_opaque] = layer_image[:,:,:3][where_opaque]
        return cast(Rgb, image)

    @classmethod
    def from_args(cls,
            height: int,
            width: int,
            layer_configs: list[LayerConfig],
            background_color: str = "#ffffff"):
        layers = [Layer.from_args(config, height, width, []) for config in layer_configs]
        return cls(height, width, layers, background_color=background_color)

    def set_sources(self, pixmap_interfaces: dict[int, list[PixmapSourceInterface]]):
        for i, layer in enumerate(self.layers):
            layer.set_sources(pixmap_interfaces.get(i, []))
