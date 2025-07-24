import numpy

from .accumulator import Accumulator
from ..utils import parse_hex_color, compose_top, compose_additive, compose_subtractive, compose_average
from ..flow import Direction


class StackAccumulator(Accumulator):

    def __init__(self, width: int, height: int, bg_color: str = "ffffff",
                 composer: str = "top", **acc_args):
        Accumulator.__init__(self, width, height, **acc_args)
        self.bg_color = parse_hex_color(bg_color)
        self.composer = {
            "top": compose_top,
            "add": compose_additive,
            "sub": compose_subtractive,
            "avg": compose_average,
        }[composer]
        self.stacks = []
        for i in range(self.height):
            self.stacks.append([])
            for j in range(self.width):
                self.stacks[i].append([(i, j)])

    def update(self, flow: numpy.ndarray, direction: Direction):
        self._update_flow(flow)
        for i in range(self.height):
            for j in range(self.width):
                mj = self.flow_int[i, j, 0]
                mi = self.flow_int[i, j, 1]
                if (mj == 0 and mi == 0):
                    continue
                if direction == Direction.FORWARD:
                    srci, srcj = i, j
                    desti = max(0, min(self.height - 1, i + mi))
                    destj = max(0, min(self.width - 1, j + mj))
                else: # Direction.BACKWARD:
                    desti, destj = i, j
                    srci = max(0, min(self.height - 1, i + mi))
                    srcj = max(0, min(self.width - 1, j + mj))
                if not self.stacks[srci][srcj]:
                    continue
                self.stacks[desti][destj].append(self.stacks[srci][srcj].pop())

    def apply(self, bitmap: numpy.ndarray) -> numpy.ndarray:
        out = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
        for i in range(self.height):
            for j in range(self.width):
                if self.stacks[i][j]:
                    out[i][j] = self.composer(*[tuple(bitmap[*xy, :3]) for xy in self.stacks[i][j]])
                else:
                    out[i][j] = self.bg_color
        return out

    def get_accumulator_array(self) -> numpy.ndarray:
        arr = numpy.zeros((self.height, self.width, 2))
        for i in range(self.height):
            for j in range(self.width):
                if self.stacks[i][j]:
                    mi, mj = self.stacks[i][j][-1]
                    arr[i, j] = [mj - j, mi - i]
        return arr