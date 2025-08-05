import os
import sys
import unittest

import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import transflow.accumulator.canvas
import transflow.accumulator.crumble
import transflow.accumulator.mapping
import transflow.accumulator.stack
import transflow.accumulator.sum
from transflow.accumulator import Accumulator


class TestAccumulator(unittest.TestCase):
    
    WIDTH = 854
    HEIGHT = 480
    NULL_FLOW = numpy.zeros((HEIGHT, WIDTH, 2)).astype(numpy.float32)
    RANDOM_FLOW = numpy.random.random((HEIGHT, WIDTH, 2)).astype(numpy.float32) * 2 - 1
    RANDOM_BITMAP = numpy.random.random((HEIGHT, WIDTH, 3)).astype(numpy.uint8)
    BLANK_BITMAP = numpy.zeros((HEIGHT, WIDTH, 3)).astype(numpy.uint8)
    MASK_PATH = "assets/Mask.png"
    
    def _test_acc(self,
            acc: Accumulator,
            same_output: bool = False,
            blank_output: bool = False):
        self.assertEqual(acc.width, self.WIDTH)
        self.assertEqual(acc.height, self.HEIGHT)
        acc.update(self.RANDOM_FLOW)
        self.assertEqual(acc.flow_int.dtype, numpy.int32)
        self.assertEqual(acc.flow_flat.shape, (self.HEIGHT * self.WIDTH,))
        output = acc.apply(self.RANDOM_BITMAP)
        self.assertEqual(output.shape, (self.HEIGHT, self.WIDTH, 3))
        self.assertEqual(output.dtype, numpy.uint8)
        if same_output:
            self.assertEqual(numpy.sum(numpy.abs(output - self.RANDOM_BITMAP)), 0)
        if blank_output:
            self.assertEqual(numpy.sum(numpy.abs(output - self.BLANK_BITMAP)), 0)
    
    def test_map(self):
        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method="map", reset_mode="off")
        self.assertIsInstance(acc, transflow.accumulator.mapping.MappingAccumulator)
        self._test_acc(acc, same_output=True)
    
    def test_sum(self):
        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method="sum", reset_mode="off")
        self.assertIsInstance(acc, transflow.accumulator.sum.SumAccumulator)
        self._test_acc(acc, same_output=True)
    
    def test_stack(self):
        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method="stack",
            reset_mode="off", bg_color="000000")
        self.assertIsInstance(acc, transflow.accumulator.stack.StackAccumulator)
        self._test_acc(acc, blank_output=True)
    
    def test_canvas(self):
        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method="canvas",
            reset_mode="off", initial_canvas="000000")
        self.assertIsInstance(acc, transflow.accumulator.canvas.CanvasAccumulator)
        self._test_acc(acc, blank_output=True)
    
    def test_crumble(self):
        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method="crumble",
            reset_mode="off", bg_color="000000")
        self.assertIsInstance(acc, transflow.accumulator.crumble.CrumbleAccumulator)
        self._test_acc(acc, blank_output=True)
    
    def test_reset(self):
        for method in ["map", "stack", "sum", "crumble", "canvas"]:
            for mode in ["linear", "random"]:
                expect_warning = (method == "crumble" and mode == "linear" or method == "canvas" and mode == "linear")
                if expect_warning:
                    with self.assertWarns(UserWarning):
                        acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method=method, reset_mode=mode, reset_mask_path=self.MASK_PATH)
                else:
                    acc = Accumulator.from_args(self.WIDTH, self.HEIGHT, method=method, reset_mode=mode, reset_mask_path=self.MASK_PATH)
                self._test_acc(acc)


if __name__ == "__main__":   
    unittest.main()