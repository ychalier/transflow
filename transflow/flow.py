import enum
import json
import math
import os
import re
import threading
import warnings
import zipfile

import cv2
import numpy

from .utils import load_mask


@enum.unique
class FlowDirection(enum.Enum):
    FORWARD = 0 # past to present
    BACKWARD = 1 # present to past


@enum.unique
class FlowMethod(enum.Enum):
    FARNEBACK = 0
    HORN_SCHUNCK = 1

    @classmethod
    def from_string(cls, string: str):
        if string == "farneback":
            return FlowMethod.FARNEBACK
        if string == "hornschunck":
            return FlowMethod.HORN_SCHUNCK
        raise ValueError(f"Invalid Flow Method: {string}")


class FlowSource:

    def __init__(self, direction: FlowDirection, mask_path: str | None = None,
                 kernel_path: str | None = None, flow_gain: str | None = None,
                 seek_ckpt: int | None = None, seek_time: float | None = None,
                 duration_time: float | None = None, repeat: int = 1):
        self.direction = direction
        self.width: int = None
        self.height: int = None
        self.framerate: float = None
        self.mask: numpy.ndarray | None = None\
            if mask_path is None else load_mask(mask_path, newaxis=True)
        self.kernel: numpy.ndarray | None = None\
            if kernel_path is None else numpy.load(kernel_path)
        self.flow_gain = None
        self.flow_gain_string = flow_gain
        self.seek_ckpt = seek_ckpt
        self.seek_time = seek_time
        self.duration_time = duration_time
        self.frame_index = 0
        self.frame_cursor = 0
        self.is_stream: bool = False
        self.length: int | None = None
        self.start_frame: int = 0
        self.end_frame: int = 0
        self.repeat = repeat

    def set_metadata(self, width: int, height: int, framerate: float, length: int):
        self.width = width
        self.height = height
        self.framerate = framerate

        if length <= 0:
            length = None

        self.is_stream = length == None
        if self.is_stream and self.repeat > 1:
            warnings.warn("Flow source is a stream, cannot repeat it!")
            self.repeat = 1
        if self.is_stream and self.seek_time is not None and self.seek_time > 0:
            warnings.warn("Flow source is a stream, seek time is ignored!")
            self.seek_time = None

        if self.seek_time is not None and not self.is_stream:
            self.start_frame = int(self.seek_time * self.framerate)
        else:
            self.start_frame = 0

        if self.duration_time is not None:
            self.end_frame = self.start_frame + int(self.duration_time * self.framerate)
            if length is not None:
                self.end_frame = min(self.end_frame, length)
        else:
            self.end_frame = length

        if self.repeat == 0:
            self.length = None
        elif self.is_stream:
            self.length = self.end_frame
        else:
            self.length = self.repeat * (self.end_frame - self.start_frame)

        real_start_frame = self.start_frame
        ckpt_start_frame = real_start_frame
        if self.seek_ckpt is not None:
            self.frame_index = self.seek_ckpt
            ckpt_start_frame += self.seek_ckpt % (self.end_frame - self.start_frame)
        self.start_frame = ckpt_start_frame
        self.rewind()
        self.start_frame = real_start_frame

    def __len__(self):
        return self.length

    def __next__(self) -> numpy.ndarray:
        if self.length is not None and self.frame_index >= self.length:
            raise StopIteration
        if self.frame_cursor == self.end_frame:
            self.rewind()
        array = self.next()
        self.frame_index += 1
        self.frame_cursor += 1
        return array

    @property
    def t(self) -> float:
        return 0 if self.framerate is None else self.frame_index / self.framerate

    def next(self) -> numpy.ndarray:
        raise NotImplementedError()

    def rewind(self):
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
                  seek_ckpt: int | None = None, seek_time: float | None = None,
                  duration_time: float | None = None, repeat: int = 1):
        if "::" in flow_path:
            avformat, file = flow_path.split("::")
        else:
            avformat, file = None, flow_path
        args = {
            "mask_path": mask_path,
            "kernel_path": kernel_path,
            "flow_gain": flow_gain,
            "seek": seek_ckpt,
            "seek_time": seek_time,
            "duration_time": duration_time,
            "repeat": repeat,
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
                 duration_time: float | None = None, repeat: int = 1):
        FlowSource.__init__(self, FlowDirection.FORWARD, mask_path, kernel_path,
                            flow_gain, seek, seek_time, duration_time, repeat)
        self.file = file
        self.avformat = avformat
        self.container = None
        self.iterator = None

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

    def rewind(self):
        self.frame_cursor = self.start_frame
        self.container.seek(0)
        for _ in range(self.frame_cursor + 1):
            next(self.iterator)

    def next(self) -> numpy.ndarray:
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
        self.inputs = {}

    def run(self):
        import PySide6.QtCore
        import PySide6.QtWidgets
        app = PySide6.QtWidgets.QApplication([])
        window = PySide6.QtWidgets.QWidget()
        layout = PySide6.QtWidgets.QVBoxLayout()
        layout.setContentsMargins(32, 32, 32, 32)

        self.inputs: dict[str, PySide6.QtWidgets.QSpinBox | PySide6.QtWidgets.QDoubleSpinBox] = {}

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

        fb_pyr_scale_input = PySide6.QtWidgets.QDoubleSpinBox()
        fb_pyr_scale_input.setDecimals(2)
        fb_pyr_scale_input.setRange(.01, .99)
        fb_pyr_scale_input.setSingleStep(.01)
        fb_pyr_scale_input.setValue(self.config.fb_pyr_scale)
        fb_pyr_scale_input.valueChanged.connect(lambda: self.callback("fb_pyr_scale", fb_pyr_scale_input))
        add_to_layout("fb_pyr_scale", fb_pyr_scale_input, "image scale (<1) to build "\
                      "pyramids for each image")
        self.inputs["fb_pyr_scale"] = fb_pyr_scale_input

        fb_levels_input = PySide6.QtWidgets.QSpinBox()
        fb_levels_input.setMinimum(1)
        fb_levels_input.setValue(self.config.fb_levels)
        fb_levels_input.valueChanged.connect(lambda: self.callback("fb_levels", fb_levels_input))
        add_to_layout("fb_levels", fb_levels_input, "number of pyramid layers "\
                      "including the initial image")
        self.inputs["fb_levels"] = fb_levels_input

        fb_winsize_input = PySide6.QtWidgets.QSpinBox()
        fb_winsize_input.setValue(self.config.fb_winsize)
        fb_winsize_input.setMinimum(1)
        fb_winsize_input.valueChanged.connect(lambda: self.callback("fb_winsize", fb_winsize_input))
        add_to_layout("fb_winsize", fb_winsize_input, "averaging window size")
        self.inputs["fb_winsize"] = fb_winsize_input

        fb_iterations_input = PySide6.QtWidgets.QSpinBox()
        fb_iterations_input.setValue(self.config.fb_iterations)
        fb_iterations_input.setMinimum(1)
        fb_iterations_input.valueChanged.connect(lambda: self.callback("fb_iterations", fb_iterations_input))
        add_to_layout("fb_iterations", fb_iterations_input, "number of iterations "\
                      "the algorithm does at each pyramid level")
        self.inputs["fb_iterations"] = fb_iterations_input

        fb_poly_n_input = PySide6.QtWidgets.QSpinBox()
        fb_poly_n_input.setValue(self.config.fb_poly_n)
        fb_poly_n_input.setMinimum(1)
        fb_poly_n_input.valueChanged.connect(lambda: self.callback("fb_poly_n", fb_poly_n_input))
        add_to_layout("fb_poly_n", fb_poly_n_input, "size of the pixel neighborhood "\
                      "used to find polynomial expansion in each pixel")
        self.inputs["fb_poly_n"] = fb_poly_n_input

        fb_poly_sigma_input = PySide6.QtWidgets.QDoubleSpinBox()
        fb_poly_sigma_input.setDecimals(2)
        fb_poly_sigma_input.setValue(self.config.fb_poly_sigma)
        fb_poly_sigma_input.valueChanged.connect(lambda: self.callback("fb_poly_sigma", fb_poly_sigma_input))
        add_to_layout("fb_poly_sigma", fb_poly_sigma_input, "standard deviation of "\
            "the Gaussian that is used to smooth derivatives used as a basis "\
            "for the polynomial expansion")
        self.inputs["fb_poly_sigma"] = fb_poly_sigma_input

        hs_alpha_input = PySide6.QtWidgets.QDoubleSpinBox()
        hs_alpha_input.setMinimum(0.001)
        hs_alpha_input.setDecimals(3)
        hs_alpha_input.setValue(self.config.hs_alpha)
        hs_alpha_input.valueChanged.connect(lambda: self.callback("hs_alpha", hs_alpha_input))
        add_to_layout("hs_alpha", hs_alpha_input, "Horn-Schunck alpha")
        self.inputs["hs_alpha"] = hs_alpha_input

        hs_iterations_input = PySide6.QtWidgets.QSpinBox()
        hs_iterations_input.setValue(self.config.hs_iterations)
        hs_iterations_input.valueChanged.connect(lambda: self.callback("hs_iterations", hs_iterations_input))
        add_to_layout("hs_iterations", hs_iterations_input, "Horn-Schunck iterations")
        self.inputs["hs_iterations"] = hs_iterations_input

        hs_decay_input = PySide6.QtWidgets.QDoubleSpinBox()
        hs_decay_input.setDecimals(3)
        hs_decay_input.setValue(self.config.hs_decay)
        hs_decay_input.valueChanged.connect(lambda: self.callback("hs_decay", hs_decay_input))
        add_to_layout("hs_decay", hs_decay_input, "Horn-Schunck decay")
        self.inputs["hs_decay"] = hs_decay_input

        hs_delta_input = PySide6.QtWidgets.QDoubleSpinBox()
        hs_delta_input.setDecimals(3)
        hs_delta_input.setValue(self.config.hs_delta)
        hs_delta_input.valueChanged.connect(lambda: self.callback("hs_delta", hs_delta_input))
        add_to_layout("hs_delta", hs_delta_input, "Horn-Schunck delta")
        self.inputs["hs_delta"] = hs_delta_input

        reset_button = PySide6.QtWidgets.QPushButton("Reset")
        reset_button.setMinimumWidth(100)
        reset_button.clicked.connect(lambda: self.reset())
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(reset_button)
        layout.addLayout(hlayout)
        
        import_button = PySide6.QtWidgets.QPushButton("Import")
        import_button.setMinimumWidth(100)
        import_button.clicked.connect(lambda: self.import_config(window))
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(import_button)
        layout.addLayout(hlayout)            
        
        export_button = PySide6.QtWidgets.QPushButton("Export")
        export_button.setMinimumWidth(100)
        export_button.clicked.connect(lambda: self.export_config(window))
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(export_button)
        layout.addLayout(hlayout)

        window.setLayout(layout)

        window.setWindowTitle("OpenCV Config")
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMaximizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMinimizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

        window.show()
        app.exec()

    def reset_inputs(self):
        for key, value in self.config.to_dict().items():
            if key not in self.inputs:
                continue
            self.inputs[key].setValue(value)

    def import_config(self, window):
        import PySide6.QtWidgets
        path = PySide6.QtWidgets.QFileDialog.getOpenFileName(
            parent=window,
            caption="Import Config",
            dir=os.getcwd(),
            filter="JSON files (*.json)",
            selectedFilter="*.json")[0]
        if path:
            other = CvFlowConfig.from_file(path)
            for key, value in other.to_dict().items():
                self.config.update(key, value)
            self.reset_inputs()

    def export_config(self, window):
        import PySide6.QtWidgets
        path = PySide6.QtWidgets.QFileDialog.getSaveFileName(
            parent=window,
            caption="Export Config",
            dir=os.getcwd(),
            filter="JSON files (*.json)",
            selectedFilter="*.json")[0]
        if path:
            self.config.to_file(path)

    def reset(self):
        self.config.reset()
        self.reset_inputs()

    def callback(self, attrname, element):
        self.config.update(attrname, element.value())


class CvFlowConfig:

    def __init__(self,
            fb_pyr_scale: float = 0.5,
            fb_levels: int = 3,
            fb_winsize: int = 15,
            fb_iterations: int = 3,
            fb_poly_n: int = 5,
            fb_poly_sigma: float = 1.2,
            fb_flags: int = 0,
            hs_alpha: float = 1,
            hs_iterations: int = 3,
            hs_decay: float = 0,
            hs_delta: float = 1,
            show_window: bool = False):
        self.fb_pyr_scale = fb_pyr_scale
        self.fb_levels = fb_levels
        self.fb_winsize = fb_winsize
        self.fb_iterations = fb_iterations
        self.fb_poly_n = fb_poly_n
        self.fb_poly_sigma = fb_poly_sigma
        self.fb_flags = fb_flags
        self.hs_alpha = hs_alpha
        self.hs_iterations = hs_iterations
        self.hs_decay = hs_decay
        self.hs_delta = hs_delta
        self.show_window = show_window
        self.window: CvFlowConfigWindow = None

    def start(self):
        if not self.show_window:
            return
        self.window = CvFlowConfigWindow(self)
        self.window.start()

    def update(self, attrname, value):
        self.__setattr__(attrname, value)
    
    def reset(self):
        self.fb_pyr_scale = 0.5
        self.fb_levels = 3
        self.fb_winsize = 15
        self.fb_iterations = 3
        self.fb_poly_n = 5
        self.fb_poly_sigma = 1.2
        self.fb_flags = 0
        self.hs_alpha = 1
        self.hs_iterations = 3
        self.hs_decay = 0.95
        self.hs_delta = 1

    def to_dict(self):
        return {
            "fb_pyr_scale": self.fb_pyr_scale,
            "fb_levels": self.fb_levels,
            "fb_winsize": self.fb_winsize,
            "fb_iterations": self.fb_iterations,
            "fb_poly_n": self.fb_poly_n,
            "fb_poly_sigma": self.fb_poly_sigma,
            "fb_flags": self.fb_flags,
            "hs_alpha": self.hs_alpha,
            "hs_iterations": self.hs_iterations,
            "hs_decay": self.hs_decay,
            "hs_delta": self.hs_delta,
        }
    
    def to_file(self, path: str):
        with open(path, "w", encoding="utf8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def from_file(cls, path: str):
        with open(path, "r", encoding="utf8") as file:
            data = json.load(file)
        return cls(**data)


def calc_optical_flow_horn_schunck(
        prev_grey: numpy.ndarray,
        next_grey: numpy.ndarray,
        flow: numpy.ndarray | None = None,
        alpha: float = 1,
        max_iters: int = 3,
        decay: float = 0,
        delta: float = 1):
    import scipy.ndimage
    a = cv2.GaussianBlur(prev_grey, (5, 5), 0)
    b = cv2.GaussianBlur(next_grey, (5, 5), 0)
    if flow is None:
        u = numpy.zeros(a.shape)
        v = numpy.zeros(a.shape)
    else:
        u = decay * flow[:,:,0]
        v = decay * flow[:,:,1]
    x_kernel = numpy.array([[1, -1], [1, -1]]) * 0.25
    y_kernel = numpy.array([[1, 1], [-1, -1]]) * 0.25
    t_kernel = numpy.ones((2, 2)) * 0.25
    avg_kernel = numpy.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]) / 12
    ex = scipy.ndimage.convolve(a, x_kernel) + scipy.ndimage.convolve(b, x_kernel)
    ey = scipy.ndimage.convolve(a, y_kernel) + scipy.ndimage.convolve(b, y_kernel)
    et = scipy.ndimage.convolve(b, t_kernel) - scipy.ndimage.convolve(a, t_kernel)
    for _ in range(max_iters):
        try:
            u_avg = scipy.ndimage.convolve(u, avg_kernel)
            v_avg = scipy.ndimage.convolve(v, avg_kernel)
            c = (ex * u_avg + ey * v_avg + et) / (4 * alpha**2 + ex**2 + ey**2)
            prev = u
            u = u_avg - ex * c
            v = v_avg - ey * c
            if delta is not None and numpy.linalg.norm(u - prev, 2) < delta:
                break
        except numpy.linalg.LinAlgError:
            # Overflows might happen
            break
    return numpy.stack([u, v], axis=-1)


class CvFlowSource(FlowSource):

    def __init__(self,
            file: str,
            config: CvFlowConfig,
            size: tuple[int, int] | None = None,
            direction: FlowDirection | None = None,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            flow_gain: str | None = None,
            seek: int | None = None,
            seek_time: float | None = None,
            duration_time: float | None = None,
            repeat: int = 1,
            method: FlowMethod = FlowMethod.HORN_SCHUNCK):
        FlowSource.__init__(self,
            direction if direction is not None else FlowDirection.FORWARD,
            mask_path, kernel_path, flow_gain, seek, seek_time, duration_time,
            repeat)
        self.file = file
        self.config = config
        self.size = size
        self.method = method
        self.capture = None
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
        self.config.start()

    def rewind(self):
        self.frame_cursor = self.start_frame
        self.capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        for i in range(self.frame_cursor + 1):
            success, frame = self.capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not open video at {self.file}")
            if i == self.frame_cursor:
                self.prev_gray = self.to_gray(frame)
        self.prev_flow = None

    def next(self) -> numpy.ndarray:
        success, frame = self.capture.read()
        if frame is None or not success:
            raise StopIteration
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
        if self.method == FlowMethod.FARNEBACK:
            flow = cv2.calcOpticalFlowFarneback(
                prev=left,
                next=right,
                flow=self.prev_flow,
                pyr_scale=self.config.fb_pyr_scale,
                levels=self.config.fb_levels,
                winsize=self.config.fb_winsize,
                iterations=self.config.fb_iterations,
                poly_n=self.config.fb_poly_n,
                poly_sigma=self.config.fb_poly_sigma,
                flags=self.config.fb_flags,
            )
        elif self.method == FlowMethod.HORN_SCHUNCK:
            flow = calc_optical_flow_horn_schunck(
                prev_grey=left,
                next_grey=right,
                flow=self.prev_flow,
                alpha=self.config.hs_alpha,
                max_iters=self.config.hs_iterations,
                decay=self.config.hs_decay,
                delta=self.config.hs_delta,
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
                 duration_time: float | None = None, repeat: int = 1):
        FlowSource.__init__(self, FlowDirection.FORWARD, mask_path, kernel_path,
                            flow_gain, seek, seek_time, duration_time, repeat)
        self.path = path
        self.archive = None

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

    def rewind(self):
        self.frame_cursor = self.start_frame

    def next(self) -> numpy.ndarray:
        with self.archive.open(f"{self.frame_cursor:09d}.npy") as file:
            flow = numpy.load(file)
        return self.apply_fx(flow)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.archive.close()
