"""Create kernel files to use for Transflow.
"""
import argparse
import pathlib

import numpy


def create_kernels(folder: str = "kernels"):
    path = pathlib.Path(folder)
    (path / "3x3").mkdir(exist_ok=True, parents=True)
    (path / "5x5").mkdir(exist_ok=True, parents=True)

    numpy.save(path / "3x3" / "identity.npy", numpy.array(([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])))

    numpy.save(path / "3x3" / "gradx.npy", numpy.array(([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])))

    numpy.save(path / "3x3" / "grady.npy", numpy.array(([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])))

    numpy.save(path / "3x3" / "gradxy.npy", numpy.array(([
        [-2, -1, 0],
        [-1, 0, 1],
        [ 0, 1, 2]
    ])))

    numpy.save(path / "3x3" / "edge-cross.npy", numpy.array(([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])))

    numpy.save(path / "3x3" / "edge-box.npy", numpy.array(([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])))

    numpy.save(path / "3x3" / "edge-corners.npy", numpy.array(([
        [1, 0, -1],
        [0, 0, 0],
        [-1, 0, 1]
    ])))

    numpy.save(path / "3x3" / "sharpen.npy", numpy.array(([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])))

    numpy.save(path / "3x3" / "blur-box.npy", (1/9) * numpy.array(([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])))

    numpy.save(path / "3x3" / "blur-gaussian.npy", (1/16) * numpy.array(([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])))

    numpy.save(path / "5x5" / "blur-gaussian.npy", (1/256) * numpy.array(([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])))

    numpy.save(path / "5x5" / "unsharp.npy", (-1/256) * numpy.array(([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=str, default="kernels", nargs="?",
        help="path to kernel folder")
    args = parser.parse_args()
    create_kernels(args.folder)


if __name__ == "__main__":
    main()