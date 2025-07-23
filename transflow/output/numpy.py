import numpy

from .zip import ZipOutput


class NumpyOutput(ZipOutput):

    def __init__(self, path: str, replace: bool = False):
        ZipOutput.__init__(self, path, replace)
        self.index = 0

    def write_array(self, array: numpy.ndarray):
        with self.archive.open(f"{self.index:09d}.npy", "w") as file:
            numpy.save(file, array)
        self.index += 1