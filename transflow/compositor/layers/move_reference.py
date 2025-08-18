from ...types import Flow
from .movement import MovementLayer
from .reference import ReferenceLayer


class MoveReferenceLayer(MovementLayer, ReferenceLayer):

    def __init__(self, *args):
        MovementLayer.__init__(self, *args)
        ReferenceLayer.__init__(self, *args)

    def update(self, flow: Flow):
        MovementLayer.update(self, flow)
        ReferenceLayer.update(self, flow)
