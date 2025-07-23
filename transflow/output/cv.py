import cv2
import numpy

from .video_output import VideoOutput


class CvVideoOutput(VideoOutput):

    WINDOW_NAME = "TransFlow"

    def __enter__(self):
        cv2.namedWindow(CvVideoOutput.WINDOW_NAME, cv2.WINDOW_NORMAL)
        return self

    def feed(self, frame: numpy.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(CvVideoOutput.WINDOW_NAME, frame)
        cv2.waitKey(1)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        cv2.destroyAllWindows()
