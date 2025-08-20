import os
import sys
import unittest
from typing import cast

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transflow.compositor import Compositor
from transflow.compositor.layers.move_reference import MoveReferenceLayer
from transflow.compositor.layers.static import StaticLayer
from transflow.compositor.layers.sum import SumLayer
from transflow.compositor.layers.introduction import IntroductionLayer
from transflow.config import LayerConfig
from transflow.types import Flow


class TestCompositor(unittest.TestCase):

    def test_basic(self):
        compositor = Compositor(1, 1, [], background_color="#ff8000")
        rgb = compositor.render()
        self.assertEqual(rgb.shape, (1, 1, 3))
        self.assertEqual(rgb.dtype, numpy.uint8)
        self.assertEqual(rgb[0,0,0], 255)
        self.assertEqual(rgb[0,0,1], 128)
        self.assertEqual(rgb[0,0,2], 0)
    
    def test_moveref(self):
        layer = MoveReferenceLayer(LayerConfig(0), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)
        self.assertEqual(layer.data[0,0,0], 1)
        self.assertEqual(layer.data[0,0,1], 0)
        self.assertEqual(layer.data[0,1,0], 1)
        self.assertEqual(layer.data[0,1,1], 1)
    
    def test_moveref_reset(self):
        layer = MoveReferenceLayer(LayerConfig(0, reset_mode="random", reset_random_factor=1), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)
        self.assertEqual(layer.data[0,0,0], 0)
        self.assertEqual(layer.data[0,0,1], 0)
        self.assertEqual(layer.data[0,1,0], 0)
        self.assertEqual(layer.data[0,1,1], 1)
    
    def test_moveref_reset_mask(self):
        layer = MoveReferenceLayer(LayerConfig(0, reset_mode="random", reset_random_factor=1, reset_mask="border-left:1"), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)
        self.assertEqual(layer.data[0,0,0], 0)
        self.assertEqual(layer.data[0,0,1], 0)
        self.assertEqual(layer.data[0,1,0], 1)
        self.assertEqual(layer.data[0,1,1], 1)
    
    def test_static(self):
        layer = StaticLayer(LayerConfig(0), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)
    
    def test_sum(self):
        layer = SumLayer(LayerConfig(0), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)
    
    def test_introduction(self):
        layer = IntroductionLayer(LayerConfig(0), 2, 3, [])
        flow = cast(Flow, numpy.array([[[0, 1], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]]]).astype(numpy.float32))
        layer.update(flow)


if __name__ == "__main__":
    unittest.main()
