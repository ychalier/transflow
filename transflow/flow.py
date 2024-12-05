import enum
import json
import math
import os
import re
import threading
import zipfile

import cv2
import numpy

from .utils import load_mask


@enum.unique
class FlowDirection(enum.Enum):
    FORWARD = 0 # past to present
    BACKWARD = 1 # present to past


class FlowSource:

    def __init__(self, direction: FlowDirection, mask_path: str | None = None,
                 kernel_path: str | None = None, flow_gain: str | None = None,
                 seek: int | None = None, seek_time: float | None = None,
                 duration_time: float | None = None):
        self.direction = direction
        self.width: int = None
        self.height: int = None
        self.framerate: float = None
        self.length: int | None = None
        self.mask: numpy.ndarray | None = None\
            if mask_path is None else load_mask(mask_path, newaxis=True)
        self.kernel: numpy.ndarray | None = None\
            if kernel_path is None else numpy.load(kernel_path)
        self.flow_gain = None
        self.flow_gain_string = flow_gain
        self.frame_index = -1
        self.seek = seek
        self.seek_time = seek_time
        self.duration_time = duration_time
        self.duration: int | None = None

    def set_metadata(self, width: int, height: int, framerate: float, length: int):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.length = length
        if self.length <= 0:
            self.length = None
        if self.seek is None:
            self.seek = 0
        if self.seek_time is not None:
            self.seek = int(self.seek_time * self.framerate)
        if self.duration_time is not None:
            self.duration = min(
                self.length - self.seek,
                max(1, int(self.duration_time * self.framerate)))
        elif self.length is not None:
            self.duration = self.length - self.seek

    def __len__(self):
        return self.length

    def __next__(self) -> numpy.ndarray:
        self.frame_index += 1
        return self.next()

    @property
    def t(self) -> float:
        return 0 if self.framerate is None else self.frame_index / self.framerate

    def next(self) -> numpy.ndarray:
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __enter__(self):
        self.setup()
        if self.flow_gain_string is not None:
            self.flow_gain = eval(f"lambda t: {self.flow_gain_string}")
        return self

    def setup(self):
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def apply_fx(self, flow: numpy.ndarray) -> numpy.ndarray:
        try:
            ff = 1 if self.flow_gain is None else self.flow_gain(self.t)
        except ZeroDivisionError:
            ff = 0
        flow *= ff
        if self.mask is None:
            flow_masked = flow
        else:
            flow_masked = numpy.multiply(self.mask, flow)
        if self.kernel is None:
            return flow_masked
        import scipy.signal
        flow_filtered_x = scipy.signal.convolve2d(
            flow_masked[:,:,0], self.kernel, mode="same", boundary="fill", fillvalue=0)
        flow_filtered_y = scipy.signal.convolve2d(
            flow_masked[:,:,1], self.kernel, mode="same", boundary="fill", fillvalue=0)
        return numpy.stack([flow_filtered_x, flow_filtered_y], axis=-1)

    @classmethod
    def from_args(cls, flow_path: str, use_mvs: bool = False,
                  mask_path: str | None = None, kernel_path: str | None = None,
                  cv_config: str | None = None, flow_gain: str | None = None,
                  size: tuple[int, int] | None = None,
                  direction: FlowDirection | None = None,
                  seek: int | None = None, seek_time: float | None = None,
                  duration_time: float | None = None):
        if "::" in flow_path:
            avformat, file = flow_path.split("::")
        else:
            avformat, file = None, flow_path
        args = {
            "mask_path": mask_path,
            "kernel_path": kernel_path,
            "flow_gain": flow_gain,
            "seek": seek,
            "seek_time": seek_time,
            "duration_time": duration_time
        }
        if file.endswith(".flow.zip"):
            return ArchiveFlowSource(file, **args)
        elif use_mvs:
            return AvFlowSource(file, avformat, **args)
        else:
            if cv_config == "window":
                config = CvFlowConfig(show_window=True)
            elif cv_config is not None and os.path.isfile(cv_config):
                config = CvFlowConfig.from_file(cv_config)
            else:
                config = CvFlowConfig()
            return CvFlowSource(file, config, size, direction, **args)


class AvFlowSource(FlowSource):

    def __init__(self, file: str, avformat: str | None = None,
                 mask_path: str | None = None, kernel_path: str | None = None,
                 flow_gain: str | None = None, seek: int | None = None,
                 seek_time: float | None = None,
                 duration_time: float | None = None):
        FlowSource.__init__(self, FlowDirection.FORWARD, mask_path, kernel_path,
                            flow_gain, seek, seek_time, duration_time)
        self.file = file
        self.avformat = avformat
        self.container = None
        self.iterator = None
        self.cursor = 0

    def setup(self):
        import av
        self.container = av.open(format=self.avformat, file=self.file)
        context = self.container.streams.video[0].codec_context
        context.export_mvs = True
        self.iterator = self.container.decode(video=0)
        first_frame = next(self.iterator)
        self.set_metadata(
            first_frame.width,
            first_frame.height,
            float(context.framerate) if context.framerate is not None else 30,
            self.container.streams.video[0].frames - 1
        )
        if self.seek is not None:
            for _ in range(self.seek):
                next(self.iterator)
                self.frame_index += 1

    def next(self) -> numpy.ndarray:
        if self.duration is not None and self.cursor >= self.duration:
            raise StopIteration
        self.cursor += 1
        frame = next(self.iterator)
        vectors = frame.side_data.get("MOTION_VECTORS")
        flow = numpy.zeros((self.height, self.width, 2), dtype=int)
        if vectors is None:
            return flow
        for mv in vectors:
            assert mv.source == -1, "Encode with bf=0 and refs=1"
            i0 = mv.src_y - mv.h // 2
            i1 = mv.src_y + mv.h // 2
            j0 = mv.src_x - mv.w // 2
            j1 = mv.src_x + mv.w // 2
            dx = mv.motion_x / mv.motion_scale
            dy = mv.motion_y / mv.motion_scale
            flow[i0:i1, j0:j1] = -dx, -dy
        return self.apply_fx(flow)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.container.close()


class CvFlowConfigWindow(threading.Thread):

    def __init__(self, config: "CvFlowConfig"):
        threading.Thread.__init__(self)
        self.config = config

    def run(self):
        import PySide6.QtCore
        import PySide6.QtWidgets
        app = PySide6.QtWidgets.QApplication([])
        window = PySide6.QtWidgets.QWidget()
        layout = PySide6.QtWidgets.QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)

        def add_to_layout(label: str, element: PySide6.QtWidgets.QWidget,
                          tooltip: str | None = None):
            element.setMinimumWidth(100)
            hlayout = PySide6.QtWidgets.QHBoxLayout()
            label_element = PySide6.QtWidgets.QLabel(label)
            if tooltip is not None:
                label_element.setToolTip(tooltip)
                element.setToolTip(tooltip)
            hlayout.addWidget(label_element)
            hlayout.addWidget(element)
            layout.addLayout(hlayout)

        pyr_scale_input = PySide6.QtWidgets.QDoubleSpinBox()
        pyr_scale_input.setDecimals(2)
        pyr_scale_input.setRange(.01, .99)
        pyr_scale_input.setSingleStep(.01)
        pyr_scale_input.setValue(self.config.pyr_scale)
        pyr_scale_input.valueChanged.connect(lambda: self.callback("pyr_scale", pyr_scale_input))
        add_to_layout("pyr_scale", pyr_scale_input, "image scale (<1) to build "\
                      "pyramids for each image")

        levels_input = PySide6.QtWidgets.QSpinBox()
        levels_input.setMinimum(1)
        levels_input.setValue(self.config.levels)
        levels_input.valueChanged.connect(lambda: self.callback("levels", levels_input))
        add_to_layout("levels", levels_input, "number of pyramid layers "\
                      "including the initial image")

        winsize_input = PySide6.QtWidgets.QSpinBox()
        winsize_input.setValue(self.config.winsize)
        winsize_input.setMinimum(1)
        winsize_input.valueChanged.connect(lambda: self.callback("winsize", winsize_input))
        add_to_layout("winsize", winsize_input, "averaging window size")

        iterations_input = PySide6.QtWidgets.QSpinBox()
        iterations_input.setValue(self.config.iterations)
        iterations_input.setMinimum(1)
        iterations_input.valueChanged.connect(lambda: self.callback("iterations", iterations_input))
        add_to_layout("iterations", iterations_input, "number of iterations "\
                      "the algorithm does at each pyramid level")

        poly_n_input = PySide6.QtWidgets.QSpinBox()
        poly_n_input.setValue(self.config.poly_n)
        poly_n_input.setMinimum(1)
        poly_n_input.valueChanged.connect(lambda: self.callback("poly_n", poly_n_input))
        add_to_layout("poly_n", poly_n_input, "size of the pixel neighborhood "\
                      "used to find polynomial expansion in each pixel")

        poly_sigma_input = PySide6.QtWidgets.QDoubleSpinBox()
        poly_sigma_input.setDecimals(2)
        poly_sigma_input.setValue(self.config.poly_sigma)
        poly_sigma_input.valueChanged.connect(lambda: self.callback("poly_sigma", poly_sigma_input))
        add_to_layout("poly_sigma", poly_sigma_input, "standard deviation of "\
            "the Gaussian that is used to smooth derivatives used as a basis "\
            "for the polynomial expansion")

        window.setLayout(layout)

        window.setWindowTitle("OpenCV Config")
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMaximizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMinimizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

        window.show()
        app.exec()

    def callback(self, attrname, element):
        self.config.update(attrname, element.value())


class CvFlowConfig:

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3,
                 winsize: int = 15, iterations: int = 3, poly_n: int = 5,
                 poly_sigma: float = 1.2, flags: int = 0, show_window: bool = False):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.show_window = show_window
        self.window: CvFlowConfigWindow = None

    def start(self):
        if not self.show_window:
            return
        self.window = CvFlowConfigWindow(self)
        self.window.start()

    def update(self, attrname, value):
        self.__setattr__(attrname, value)

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r", encoding="utf8") as file:
            data = json.load(file)
        return cls(**data)


class CvFlowSource(FlowSource):

    def __init__(self, file: str, config: CvFlowConfig,
                 size: tuple[int, int] | None = None,
                 direction: FlowDirection | None = None,
                 mask_path: str | None = None,
                 kernel_path: str | None = None,
                 flow_gain: str | None = None,
                 seek: int | None = None, seek_time: float | None = None,
                 duration_time: float | None = None):
        FlowSource.__init__(self,
            direction if direction is not None else FlowDirection.FORWARD,
            mask_path, kernel_path, flow_gain, seek, seek_time, duration_time)
        self.file = file
        self.config = config
        self.size = size
        self.capture = None
        self.cursor = 0
        self.prev_gray = None
        self.prev_flow = None

    def to_gray(self, frame: numpy.ndarray) -> numpy.ndarray:
        return cv2.cvtColor(
            cv2.resize(
                frame,
                dsize=(self.width, self.height),
                interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2GRAY)

    def setup(self):
        if re.match(r"\d+", self.file):
            # file argument is probably a webcam index
            self.capture = cv2.VideoCapture(int(self.file))
        else:
            self.capture = cv2.VideoCapture(self.file)
        if self.size is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.set_metadata(
            int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(self.capture.get(cv2.CAP_PROP_FPS)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        )
        success, frame = self.capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not open video at {self.file}")
        self.prev_gray = self.to_gray(frame)
        self.prev_flow = None
        self.cursor = 1
        self.config.start()
        if self.seek is not None:
            for i in range(self.seek):
                success, frame = self.capture.read()
                if not success or frame is None:
                    raise RuntimeError(f"Could not seek past frame {i}")
                if i == self.seek - 1:
                    self.prev_gray = self.to_gray(frame)
                self.frame_index += 1
                self.cursor += 1

    def next(self) -> numpy.ndarray:
        if self.length is not None and self.cursor > self.length:
            raise StopIteration
        if self.duration is not None and self.cursor > self.seek + self.duration:
            raise StopIteration
        success, frame = self.capture.read()
        if frame is None or not success:
            raise StopIteration
        self.cursor += 1
        gray = cv2.cvtColor(
            cv2.resize(frame, dsize=(self.width, self.height),
                       interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2GRAY)
        if self.direction == FlowDirection.FORWARD:
            left, right = self.prev_gray, gray
        elif self.direction == FlowDirection.BACKWARD:
            left, right = gray, self.prev_gray
        else:
            raise ValueError(f"Invalid flow direction '{self.direction}'")
        flow = cv2.calcOpticalFlowFarneback(
            prev=left,
            next=right,
            flow=self.prev_flow,
            pyr_scale=self.config.pyr_scale,
            levels=self.config.levels,
            winsize=self.config.winsize,
            iterations=self.config.iterations,
            poly_n=self.config.poly_n,
            poly_sigma=self.config.poly_sigma,
            flags=self.config.flags,
        )
        self.prev_gray = gray
        self.prev_flow = flow
        return self.apply_fx(flow)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.capture.release()


class ArchiveFlowSource(FlowSource):

    def __init__(self, path: str, mask_path: str | None = None,
                 kernel_path: str | None = None,
                 flow_gain: str | None = None, seek: int | None = None,
                 seek_time: float | None = None,
                 duration_time: float | None = None):
        FlowSource.__init__(self, FlowDirection.FORWARD, mask_path, kernel_path,
                            flow_gain, seek, seek_time, duration_time)
        self.path = path
        self.archive = None
        self.index = 0

    def setup(self):
        self.archive = zipfile.ZipFile(self.path)
        with self.archive.open("meta.json") as file:
            data = json.loads(file.read().decode())
        # for backward compatibility, previous flows were only forward
        self.direction = FlowDirection(data.get("direction", FlowDirection.FORWARD.value))
        self.set_metadata(
            data["width"],
            data["height"],
            data["framerate"],
            len(self.archive.infolist()) - 1
        )
        if self.seek is not None:
            self.index = self.seek

    def next(self) -> numpy.ndarray:
        if self.index >= self.length:
            raise StopIteration
        if self.index >= self.seek + self.duration:
            raise StopIteration
        with self.archive.open(f"{self.index:09d}.npy") as file:
            flow = numpy.load(file)
        self.index += 1
        return self.apply_fx(flow)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.archive.close()
