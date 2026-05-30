from typing import cast, TYPE_CHECKING

import numpy

from ...config import LayerConfig
from ...utils import load_float_mask
from ...types import Flow, Rgba, FloatMask
from ..pixmap_source_interface import PixmapSourceInterface
if TYPE_CHECKING:
    from ..compositor import Compositor


class Layer:

    def __init__(self,
            compositor: 'Compositor',
            config: LayerConfig,
            height: int,
            width: int,
            sources: list[PixmapSourceInterface],
            ):
        self.compositor = compositor
        self.config = config
        self.height = height
        self.width = width
        self.sources: list[PixmapSourceInterface] = sources
        self.rgba = numpy.zeros((self.height, self.width, 4), dtype=numpy.uint8)
        self.mask_alpha: FloatMask = load_float_mask(self.config.mask_alpha, (self.height, self.width), 1)
    
    def set_sources(self, sources: list[PixmapSourceInterface]):
        self.sources = sources

    def update(self, flow: Flow):
        raise NotImplementedError()

    def render(self) -> Rgba:
        self.rgba[:,:,3] = self.mask_alpha * self.rgba[:,:,3]
        return cast(Rgba, numpy.clip(self.rgba, 0, 255).astype(numpy.uint8))

    @classmethod
    def from_args(cls,
            compositor: 'Compositor',
            config: LayerConfig,
            height: int,
            width: int,
            sources: list[PixmapSourceInterface]):
        args = [compositor, config, height, width, sources]
        if config.classname == "moveref":
            from .move_reference import MoveReferenceLayer
            return MoveReferenceLayer(*args)
        if config.classname == "introduction":
            from .introduction import IntroductionLayer
            return IntroductionLayer(*args)
        if config.classname == "static":
            from .static import StaticLayer
            return StaticLayer(*args)
        if config.classname == "sum":
            from .sum import SumLayer
            return SumLayer(*args)
        raise ValueError(f"Unknown layer classname {config.classname}")
