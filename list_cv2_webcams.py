import argparse
import math
import os
import time

import cv2
import numpy

cv2.setLogLevel(1)


def scan_webcam(device_index: int, thumbnail_size: tuple[int, int]) -> cv2.Mat | None:
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = capture.get(cv2.CAP_PROP_FPS)
    success, frame = capture.read()
    if not success:
        capture.release()
        return None
    capture.release()
    res_str = f"{width}*{height}"
    fps_str = f"{framerate:.2f} fps"
    print(f"Device #{device_index}: {res_str:>9} {fps_str:>10}")
    thumbnail = cv2.resize(frame, thumbnail_size)
    fontargs = cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2
    cv2.putText(thumbnail, f"Device: {device_index}", (4, 24), *fontargs)
    cv2.putText(thumbnail, f"{width}*{height}", (4, 54), *fontargs)
    cv2.putText(thumbnail, f"{framerate:.2f} FPS", (4, 84), *fontargs)
    return thumbnail


def scan_webcams(max_tries: int, thumbnail_size: tuple[int, int]) -> numpy.ndarray:
    thumbnails: list[cv2.Mat] = []
    for device_index in range(max_tries):
        thumbnail = scan_webcam(device_index, thumbnail_size)
        if thumbnail is not None:
            thumbnails.append(thumbnail)
    ncols = math.ceil(math.sqrt(len(thumbnails)))
    nrows = math.ceil(len(thumbnails) / ncols)
    rows = []
    for i in range(nrows):
        arrays = thumbnails[i*ncols:(i+1)*ncols]
        while len(arrays) < ncols:
            arrays.append(numpy.zeros(thumbnails[0].shape, dtype=numpy.uint8))
        rows.append(numpy.concatenate(arrays, axis=1))
    grid = numpy.concatenate(rows, axis=0)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tw", "--thumbnail-width", type=int, default=320)
    parser.add_argument("-th", "--thumbnail-height", type=int, default=240)
    parser.add_argument("-m", "--max-tries", type=int, default=10)
    parser.add_argument("-o", "--output-path", type=str, default="cv2.jpg")
    args = parser.parse_args()
    grid = scan_webcams(args.max_tries, (args.thumbnail_width, args.thumbnail_height))
    cv2.imwrite(args.output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    os.startfile(args.output_path)
    time.sleep(1)


if __name__ == "__main__":
    main()