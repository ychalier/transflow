import numpy

from ..utils import parse_lambda_expression
from ..types import Flow


class FlowFilter:

    def __init__(self):
        pass

    def apply(self, flow: Flow, t: float) -> None:
        raise NotImplementedError()
    
    @classmethod
    def from_args(cls, filter_name: str, filter_args: tuple[str, ...]):
        if filter_name == "scale":
            if len(filter_args) != 1:
                raise ValueError(f"Invalid number of arguments: {filter_name} {filter_args}")
            return ScaleFlowFilter(filter_args)
        if filter_name == "threshold":
            if len(filter_args) != 1:
                raise ValueError(f"Invalid number of arguments: {filter_name} {filter_args}")
            return ThresholdFlowFilter(filter_args)
        if filter_name == "clip":
            if len(filter_args) != 1:
                raise ValueError(f"Invalid number of arguments: {filter_name} {filter_args}")
            return ClipFlowFilter(filter_args)
        if filter_name == "polar":
            if len(filter_args) != 2:
                raise ValueError(f"Invalid number of arguments: {filter_name} {filter_args}")
            return PolarFlowFilter(filter_args)
        raise ValueError(f"Unknown filter name '{filter_name}'")


class ScaleFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: Flow, t: float):
        flow *= self.expr(t)


class ThresholdFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: Flow, t: float):
        height, width, _ = flow.shape
        norm = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        threshold = self.expr(t)
        where = numpy.where(norm <= threshold)
        flow[where] = 0


class ClipFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: Flow, t: float):
        height, width, _ = flow.shape
        norm = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        factors = numpy.ones((height, width))
        threshold = self.expr(t)
        where = numpy.where(norm >= threshold)
        factors[where] = threshold / norm[where]
        flow[:,:,0] *= factors
        flow[:,:,1] *= factors


class PolarFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str, str]):
        self.expr_radius = parse_lambda_expression(filter_args[0], ("t", "r", "a"))
        self.expr_theta = parse_lambda_expression(filter_args[1], ("t", "r", "a"))
    
    def apply(self, flow: Flow, t: float):
        height, width, _ = flow.shape
        radius = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        theta = numpy.atan2(flow[:,:,1], flow[:,:,0])
        new_radius = self.expr_radius(t, radius, theta)
        new_theta = self.expr_theta(t, radius, theta)
        flow[:,:,1] = new_radius * numpy.sin(new_theta)
        flow[:,:,0] = new_radius * numpy.cos(new_theta)
