import cv2
import numpy

from .video_output import VideoOutput


cv_video_outputs = []
def on_mouse(event, x, y, flags, userdata):
    # x,y are window coords == image coords when WINDOW_AUTOSIZE
    global cv_video_outputs
    if event == cv2.EVENT_MOUSEMOVE:
        userdata["mouse_pos"] = (x, y)
        for cv_video_output in cv_video_outputs:
            cv_video_output.draw()


class CvVideoOutput(VideoOutput):

    WINDOW_NAME = "TransFlow"
    
    def __init__(self, *args, show_mouse_cursor: bool = False, **kwargs):
        VideoOutput.__init__(self, *args, **kwargs)
        self.state = {"mouse_pos": (-1, -1)}
        self.frame = numpy.zeros((1, 1, 3), dtype=numpy.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_mouse_cursor = show_mouse_cursor

    def __enter__(self):
        cv2.namedWindow(CvVideoOutput.WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.show_mouse_cursor:
            cv2.setMouseCallback(CvVideoOutput.WINDOW_NAME, on_mouse, self.state)
            global cv_video_outputs
            cv_video_outputs.append(self)
        return self

    def feed(self, frame: numpy.ndarray):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.draw()
    
    def draw(self):
        frame = self.frame.copy()
        if self.show_mouse_cursor:
            x, y = self.state["mouse_pos"]
            pos = (4, 10)
            b, g, r = frame[y, x].tolist()
            text = f"({x}, {y})  RGB=({r}, {g}, {b})"
            cv2.putText(frame, text, pos, self.font, 0.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, pos, self.font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(CvVideoOutput.WINDOW_NAME, frame)
        cv2.waitKey(1)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        cv2.destroyAllWindows()
