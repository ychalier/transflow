import unittest

import numpy

import transflow.accumulator as ta


class TestAccumulator(unittest.TestCase):
    
    WIDTH = 854
    HEIGHT = 480
    NULL_FLOW = numpy.zeros((HEIGHT, WIDTH, 2)).astype(numpy.float32)
    RANDOM_FLOW = numpy.random.random((HEIGHT, WIDTH, 2)).astype(numpy.float32) * 2 - 1
    RANDOM_BITMAP = numpy.random.random((HEIGHT, WIDTH, 3)).astype(numpy.uint8)
    BLANK_BITMAP = numpy.zeros((HEIGHT, WIDTH, 3)).astype(numpy.uint8)
    MASK_PATH = "assets/Mask.png"
    
    def _test_acc(self,
            acc: ta.Accumulator,
            direction: ta.FlowDirection = ta.FlowDirection.BACKWARD,
            same_output: bool = False,
            blank_output: bool = False):
        self.assertEqual(acc.width, self.WIDTH)
        self.assertEqual(acc.height, self.HEIGHT)
        acc.update(self.RANDOM_FLOW, direction)
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
        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method="map", reset_mode="off")
        self.assertIsInstance(acc, ta.MappingAccumulator)
        self._test_acc(acc, ta.FlowDirection.BACKWARD, same_output=True)
        self._test_acc(acc, ta.FlowDirection.FORWARD, same_output=True)
    
    def test_sum(self):
        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method="sum", reset_mode="off")
        self.assertIsInstance(acc, ta.SumAccumulator)
        self._test_acc(acc, ta.FlowDirection.BACKWARD, same_output=True)
        with self.assertWarns(UserWarning):
            self._test_acc(acc, ta.FlowDirection.FORWARD, same_output=True)
    
    def test_stack(self):
        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method="stack",
            reset_mode="off", bg_color="000000")
        self.assertIsInstance(acc, ta.StackAccumulator)
        self._test_acc(acc, ta.FlowDirection.BACKWARD, blank_output=True)
        self._test_acc(acc, ta.FlowDirection.FORWARD, blank_output=True)
    
    def test_canvas(self):
        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method="canvas",
            reset_mode="off", initial_canvas="000000")
        self.assertIsInstance(acc, ta.CanvasAccumulator)
        self._test_acc(acc, ta.FlowDirection.BACKWARD, blank_output=True)
        self._test_acc(acc, ta.FlowDirection.FORWARD, blank_output=True)
    
    def test_crumble(self):
        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method="crumble",
            reset_mode="off", bg_color="000000")
        self.assertIsInstance(acc, ta.CrumbleAccumulator)
        self._test_acc(acc, ta.FlowDirection.BACKWARD, blank_output=True)
        self._test_acc(acc, ta.FlowDirection.FORWARD, blank_output=True)
    
    def test_reset(self):
        for method in ["map", "stack", "sum", "crumble", "canvas"]:
            for mode in ["linear", "random"]:
                expect_warning = (method == "crumble" and mode == "linear" or method == "canvas" and mode == "linear")
                if expect_warning:
                    with self.assertWarns(UserWarning):
                        acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method=method, reset_mode=mode, reset_mask_path=self.MASK_PATH)
                else:
                    acc = ta.Accumulator.from_args(self.WIDTH, self.HEIGHT, method=method, reset_mode=mode, reset_mask_path=self.MASK_PATH)
                self._test_acc(acc)
