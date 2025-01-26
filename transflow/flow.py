import enum
import json
import math
import os
import random
import re
import threading
import warnings
import zipfile

import cv2
import numpy

from .utils import load_mask, parse_lambda_expression


@enum.unique
class FlowDirection(enum.Enum):
    FORWARD = 0 # past to present
    BACKWARD = 1 # present to past


@enum.unique
class FlowMethod(enum.Enum):
    FARNEBACK = 0
    HORN_SCHUNCK = 1
    LUKAS_KANADE = 2
    LITEFLOWNET = 3

    @classmethod
    def from_string(cls, string: str):
        if string == "farneback":
            return FlowMethod.FARNEBACK
        if string == "horn-schunck":
            return FlowMethod.HORN_SCHUNCK
        if string == "lukas-kanade":
            return FlowMethod.LUKAS_KANADE
        if string == "liteflownet":
            return FlowMethod.LITEFLOWNET
        raise ValueError(f"Invalid Flow Method: {string}")

    @staticmethod
    def to_string(method: "FlowMethod"):
        if method == FlowMethod.FARNEBACK:
            return "farneback"
        if method == FlowMethod.HORN_SCHUNCK:
            return "horn-schunck"
        if method == FlowMethod.LUKAS_KANADE:
            return "lukas-kanade"
        if method == FlowMethod.LITEFLOWNET:
            return "liteflownet"
        raise ValueError(f"Unknown flow method {method}")


@enum.unique
class LockMode(enum.Enum):
    STAY = 0
    SKIP = 1

    @classmethod
    def from_arg(cls, arg):
        if isinstance(arg, LockMode):
            return arg
        assert isinstance(arg, str)
        if arg == "stay":
            return LockMode.STAY
        if arg == "skip":
            return LockMode.SKIP
        raise ValueError(f"Invalid Lock Mode: {arg}")


class FlowFilter:

    def __init__(self):
        pass

    def apply(self, flow: numpy.ndarray, t: float) -> None:
        raise NotImplementedError()
    
    @classmethod
    def from_args(cls, filter_name: str, filter_args: tuple[str]):
        if filter_name == "scale":
            return ScaleFlowFilter(filter_args)
        if filter_name == "threshold":
            return ThresholdFlowFilter(filter_args)
        if filter_name == "clip":
            return ClipFlowFilter(filter_args)
        if filter_name == "polar":
            return PolarFlowFilter(filter_args)
        raise ValueError(f"Unknown filter name '{filter_name}'")


class ScaleFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: numpy.ndarray, t: float):
        flow *= self.expr(t)


class ThresholdFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: numpy.ndarray, t: float):
        height, width, _ = flow.shape
        norm = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        threshold = self.expr(t)
        where = numpy.where(norm <= threshold)
        flow[where] = 0


class ClipFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr = parse_lambda_expression(filter_args[0])
    
    def apply(self, flow: numpy.ndarray, t: float):
        height, width, _ = flow.shape
        norm = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        factors = numpy.ones((height, width))
        threshold = self.expr(t)
        where = numpy.where(norm >= threshold)
        factors[where] = threshold / norm[where]
        flow[:,:,0] *= factors
        flow[:,:,1] *= factors


class PolarFlowFilter(FlowFilter):

    def __init__(self, filter_args: tuple[str]):
        self.expr_radius = parse_lambda_expression(filter_args[0], ("t", "r", "a"))
        self.expr_theta = parse_lambda_expression(filter_args[1], ("t", "r", "a"))
    
    def apply(self, flow: numpy.ndarray, t: float):
        height, width, _ = flow.shape
        radius = numpy.linalg.norm(flow.reshape(height * width, 2), axis=1).reshape((height, width))
        theta = numpy.atan2(flow[:,:,1], flow[:,:,0])
        new_radius = self.expr_radius(t, radius, theta)
        new_theta = self.expr_theta(t, radius, theta)
        flow[:,:,1] = new_radius * numpy.sin(new_theta)
        flow[:,:,0] = new_radius * numpy.cos(new_theta)


class FlowSource:

    def __init__(self,
            direction: FlowDirection,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            flow_filters: str | None = None,
            seek_ckpt: int | None = None,
            seek_time: float | None = None,
            duration_time: float | None = None,
            repeat: int = 1,
            lock_expr: str | None = None,
            lock_mode: str | LockMode = LockMode.STAY):
        self.direction = direction
        self.width: int = None
        self.height: int = None
        self.framerate: float = None
        self.mask: numpy.ndarray | None = None\
            if mask_path is None else load_mask(mask_path, newaxis=True)
        self.kernel: numpy.ndarray | None = None\
            if kernel_path is None else numpy.load(kernel_path)
        self.flow_filters = None
        self.flow_filters_string = flow_filters
        self.seek_ckpt = seek_ckpt
        self.seek_time = seek_time
        self.duration_time = duration_time
        self.input_frame_index = 0
        self.output_frame_index = 0
        self.is_stream: bool = False
        self.length: int | None = None
        self.start_frame: int = 0
        self.end_frame: int = 0
        self.repeat = repeat
        self.prev_flow = None
        self.lock_expr = lock_expr
        self.lock_mode = LockMode.from_arg(lock_mode)
        self.lock_start = None

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

        if self.length is not None and self.lock_mode == LockMode.STAY:
            for _, lock_duration in self.lock_expr:
                self.length += int(lock_duration * framerate)

        real_start_frame = self.start_frame
        ckpt_start_frame = real_start_frame
        if self.seek_ckpt is not None:
            self.output_frame_index = self.seek_ckpt
            ckpt_start_frame += self.seek_ckpt % (self.end_frame - self.start_frame)
        self.start_frame = ckpt_start_frame
        self.rewind()
        self.start_frame = real_start_frame

    def __len__(self):
        return self.length
    
    def read_next_flow(self) -> numpy.ndarray:
        if self.input_frame_index == self.end_frame:
            self.rewind()
        flow = self.next()
        self.input_frame_index += 1
        return flow

    def __next__(self) -> numpy.ndarray:
        if self.length is not None and self.output_frame_index >= self.length:
            raise StopIteration
        locked = False
        if self.lock_expr is not None:
            if self.lock_mode == LockMode.STAY:
                was_locked = self.lock_start is not None
                if was_locked:
                    lock_elapsed = self.t - self.lock_start
                    locked = lock_elapsed < self.lock_expr[0][1]
                    if not locked:
                        self.lock_expr.pop(0)
                        self.lock_start = None
                if self.lock_expr and ((not was_locked) or (not locked)):
                    locked = self.t >= self.lock_expr[0][0]
                    if locked:
                        self.lock_start = self.t
            elif self.lock_mode == LockMode.SKIP:
                locked = self.lock_expr(self.t)            
        flow = self.prev_flow if locked else self.read_next_flow()
        self.prev_flow = flow
        if locked and self.lock_mode == LockMode.SKIP:
            self.read_next_flow()
        self.output_frame_index += 1
        return self.post_process(flow)

    @property
    def t(self) -> float:
        return 0 if self.framerate is None else self.output_frame_index / self.framerate

    def next(self) -> numpy.ndarray:
        raise NotImplementedError()

    def rewind(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __enter__(self):
        self.setup()
        return self

    def setup(self):
        if self.lock_expr is not None:
            assert isinstance(self.lock_expr, str)
            if self.lock_mode == LockMode.STAY:
                if "(" not in self.lock_expr:
                    self.lock_expr = f"({self.lock_expr})"
                self.lock_expr = eval(f"[{self.lock_expr},]")
            elif self.lock_mode == LockMode.SKIP:
                self.lock_expr = parse_lambda_expression(self.lock_expr)
        if self.flow_filters_string is not None:
            self.flow_filters: list[FlowFilter] = []
            for filter_string in self.flow_filters_string.strip().split(";"):
                if filter_string.split() == "":
                    continue
                i = filter_string.index("=")
                self.flow_filters.append(FlowFilter.from_args(
                    filter_string[:i].strip(),
                    filter_string[i+1:].strip().split(":")))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def post_process(self, flow: numpy.ndarray) -> numpy.ndarray:
        if self.flow_filters:
            for flow_filter in self.flow_filters:
                flow_filter.apply(flow, self.t)
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
    def from_args(cls,
            flow_path: str,
            use_mvs: bool = False,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            cv_config: str | None = None,
            flow_filters: str | None = None,
            size: tuple[int, int] | None = None,
            direction: FlowDirection | None = None,
            seek_ckpt: int | None = None,
            seek_time: float | None = None,
            duration_time: float | None = None,
            repeat: int = 1,
            lock_expr: str | None = None,
            lock_mode: str | LockMode = LockMode.STAY):
        if "::" in flow_path:
            avformat, file = flow_path.split("::")
        else:
            avformat, file = None, flow_path
        args = {
            "mask_path": mask_path,
            "kernel_path": kernel_path,
            "flow_filters": flow_filters,
            "seek_ckpt": seek_ckpt,
            "seek_time": seek_time,
            "duration_time": duration_time,
            "repeat": repeat,
            "lock_expr": lock_expr,
            "lock_mode": lock_mode
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

    def __init__(self, file: str, avformat: str | None = None, **src_args):
        FlowSource.__init__(self, FlowDirection.FORWARD, **src_args)
        self.file = file
        self.avformat = avformat
        self.container = None
        self.iterator = None

    def setup(self):
        FlowSource.setup(self)
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
        return flow

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.container.close()


class CvFlowConfigWindow(threading.Thread):

    TEMPLATE = [
        {
            "name": "farneback",
            "params": [
                {
                    "name": "fb_pyr_scale",
                    "label": "Pyr Scale",
                    "help": "image scale (<1) to build pyramids for each image",
                    "type": float,
                    "decimals": 2,
                    "min": 0.01,
                    "max": 0.99
                },
                {
                    "name": "fb_levels",
                    "label": "Levels",
                    "help": "number of pyramid layers including the initial image",
                    "type": int,
                    "min": 1,
                },
                {
                    "name": "fb_iterations",
                    "label": "Iterations",
                    "help": "number of iterations the algorithm does at each pyramid level",
                    "type": int,
                    "min": 1,
                },
                {
                    "name": "fb_poly_n",
                    "label": "Poly N",
                    "help": "size of the pixel neighborhood used to find polynomial expansion in each pixel",
                    "type": int,
                    "min": 1,
                },
                {
                    "name": "fb_poly_sigma",
                    "label": "Poly Sigma",
                    "help": "standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion",
                    "type": float,
                    "decimals": 2,
                },
            ]
        },
        {
            "name": "horn-schunck",
            "params": [
                {
                    "name": "hs_alpha",
                    "label": "Alpha",
                    "help": "regularization constant; larger values lead to smoother flow",
                    "type": float,
                    "decimals": 3,
                    "min": 0.001,
                },
                {
                    "name": "hs_iterations",
                    "label": "Max Iterations",
                    "help": "maximum number of iterations the algorithm does; may stop earlier if convergence is achieved (see `hs_delta` parameter); large value (>100) required for precise computations",
                    "type": int,
                    "min": 0,
                    "max": 1000
                },
                {
                    "name": "hs_decay",
                    "label": "Decay",
                    "help": "initial flow estimation (before any iteration) is based on previous flow scaled by hs_decay; set hs_decay=0 for no initialization; set hs_decay=1 for re-using whole previous flow; set hs_decay=0.95 for a geometric decay; using hs_decay>0 introduces an inertia effect",
                    "type": float,
                    "decimals": 3,
                },
                {
                    "name": "hs_delta",
                    "label": "Delta",
                    "help": "convergence threshold; stops when the L2 norm of the difference of the flows between two consecutive iterations drops below",
                    "type": float,
                    "decimals": 2,
                },
            ]
        },
        {
            "name": "lukas-kanade",
            "params": [
                {
                    "name": "lk_window_size",
                    "label": "Window Size",
                    "help": "size of the search window at each pyramid level",
                    "type": int,
                    "min": 1
                },
                {
                    "name": "lk_max_level",
                    "label": "Max Level",
                    "help": "0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have but no more than maxLevel",
                    "type": int,
                    "min": 0
                },
                {
                    "name": "lk_step",
                    "label": "Step",
                    "help": "size of macroblocks for estimating the flow; set lk_step=1 for a dense flow; set lk_step=16 for 16*16 macroblocks",
                    "type": int,
                    "min": 1
                }
            ]
        },
        {
            "name": "liteflownet",
            "params": []
        }
    ]

    def __init__(self, config: "CvFlowConfig"):
        threading.Thread.__init__(self)
        self.config = config
        self.inputs = {}
        self.widgets = {}

    def run(self):
        import PySide6.QtCore
        import PySide6.QtWidgets
        app = PySide6.QtWidgets.QApplication([])
        window = PySide6.QtWidgets.QWidget()
        layout_main = PySide6.QtWidgets.QVBoxLayout()
        layout_main.setContentsMargins(32, 32, 32, 32)

        self.inputs: dict[str, PySide6.QtWidgets.QSpinBox | PySide6.QtWidgets.QDoubleSpinBox] = {}
        self.widgets: dict[str, PySide6.QtWidgets.QWidget] = {}

        def add_to_layout(
                layout: PySide6.QtWidgets.QVBoxLayout,
                label: str,
                element: PySide6.QtWidgets.QWidget,
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
        
        self.method_input = PySide6.QtWidgets.QComboBox()
        self.method_input.currentIndexChanged.connect(self.callback)
        add_to_layout(layout_main, "method", self.method_input, "method")

        for method in self.TEMPLATE:
            self.method_input.addItem(method["name"])
            layout = PySide6.QtWidgets.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            widget = PySide6.QtWidgets.QWidget()
            widget.setLayout(layout)
            layout_main.addWidget(widget)
            self.widgets[method["name"]] = widget
            for param in method["params"]:
                if param["type"] == int:
                    widget_element = PySide6.QtWidgets.QSpinBox()
                elif param["type"] == float:
                    widget_element = PySide6.QtWidgets.QDoubleSpinBox()
                    widget_element.setDecimals(param["decimals"])
                    widget_element.setSingleStep(1 / (10 ** param["decimals"]))
                else:
                    raise ValueError(f"Invalid parameter type {param['type']}")
                if "min" in param:
                    widget_element.setMinimum(param["min"])
                if "max" in param:
                    widget_element.setMaximum(param["max"])
                widget_element.setValue(getattr(self.config, param["name"]))
                widget_element.valueChanged.connect(self.callback)
                add_to_layout(layout, param["label"], widget_element, param["help"])
                self.inputs[param["name"]] = widget_element

        reset_button = PySide6.QtWidgets.QPushButton("Reset")
        reset_button.setMinimumWidth(100)
        reset_button.clicked.connect(lambda: self.reset())
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(reset_button)
        layout_main.addLayout(hlayout)
        
        import_button = PySide6.QtWidgets.QPushButton("Import")
        import_button.setMinimumWidth(100)
        import_button.clicked.connect(lambda: self.import_config(window))
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(import_button)
        layout_main.addLayout(hlayout)            
        
        export_button = PySide6.QtWidgets.QPushButton("Export")
        export_button.setMinimumWidth(100)
        export_button.clicked.connect(lambda: self.export_config(window))
        hlayout = PySide6.QtWidgets.QHBoxLayout()
        hlayout.addWidget(export_button)
        layout_main.addLayout(hlayout)

        self.callback()
        window.setLayout(layout_main)

        window.setWindowTitle("OpenCV Config")
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMaximizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowMinimizeButtonHint, False)
        window.setWindowFlag(PySide6.QtCore.Qt.WindowType.WindowStaysOnTopHint, True)

        window.show()
        app.exec()

    def set_inputs_from_config(self):
        for key, value in self.config.to_dict().items():
            if key == "method":
                self.method_input.setCurrentIndex([m["name"] for m in self.TEMPLATE].index(value))
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
            selectedFilter="JSON files (*.json)")[0]
        if path:
            other = CvFlowConfig.from_file(path)
            for key, value in other.to_dict().items():
                self.config.update(key, value)
            self.set_inputs_from_config()

    def export_config(self, window):
        import PySide6.QtWidgets
        path = PySide6.QtWidgets.QFileDialog.getSaveFileName(
            parent=window,
            caption="Export Config",
            dir=os.getcwd(),
            filter="JSON files (*.json)",
            selectedFilter="JSON files (*.json)")[0]
        if path:
            self.config.to_file(path)

    def reset(self):
        self.config.reset()
        self.set_inputs_from_config()

    def callback(self):
        method_name = self.TEMPLATE[self.method_input.currentIndex()]["name"]
        method = FlowMethod.from_string(method_name)
        self.config.update("method", method)
        for widget_name, widget in self.widgets.items():
            widget.setVisible(method_name == widget_name)
        for name, widget in self.inputs.items():
            self.config.update(name, widget.value())


class CvFlowConfig:

    def __init__(self,
            method: str | FlowMethod = "farneback",
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
            lk_window_size: int = 15,
            lk_max_level: int = 2,
            lk_step: int = 1,
            show_window: bool = False):
        self.method = FlowMethod.from_string(method) if isinstance(method, str) else method
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
        self.lk_window_size = lk_window_size
        self.lk_max_level = lk_max_level
        self.lk_step = lk_step
        self.show_window = show_window
        self.window: CvFlowConfigWindow = None

    def start(self):
        if not self.show_window:
            return
        self.window = CvFlowConfigWindow(self)
        self.window.start()

    def update(self, attrname, value):
        if attrname == "method" and isinstance(value, str):
            value = FlowMethod.from_string(value)
        self.__setattr__(attrname, value)
    
    def reset(self):
        self.method = FlowMethod.FARNEBACK
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
        self.lk_window_size = 15
        self.lk_max_level = 2
        self.lk_step = 1

    def to_dict(self):
        return {
            "method": FlowMethod.to_string(self.method),
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
            "lk_window_size": self.lk_window_size,
            "lk_max_level": self.lk_max_level,
            "lk_step": self.lk_step
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
    a = cv2.GaussianBlur(prev_grey.astype(numpy.float64), (5, 5), 0)
    b = cv2.GaussianBlur(next_grey.astype(numpy.float64), (5, 5), 0)
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
        u_avg = scipy.ndimage.convolve(u, avg_kernel)
        v_avg = scipy.ndimage.convolve(v, avg_kernel)
        c = numpy.divide(
            numpy.multiply(ex, u_avg) + numpy.multiply(ey, v_avg) + et,
            alpha ** 2 + numpy.pow(ex, 2) + numpy.pow(ey, 2)
        )
        prev = u
        u = u_avg - numpy.multiply(ex, c)
        v = v_avg - numpy.multiply(ey, c)
        if delta is not None and numpy.linalg.norm(u - prev, 2) < delta:
            break
    return numpy.stack([u, v], axis=-1)


def calc_optical_flow_lukas_kanade(
        prev_grey: numpy.ndarray,
        next_grey: numpy.ndarray,
        win_size: int,
        max_level: int,
        step: int) -> numpy.ndarray:
    m, n = prev_grey.shape
    p0 = numpy.stack(
        numpy.meshgrid(
            numpy.arange(0, prev_grey.shape[1], step),
            numpy.arange(0, prev_grey.shape[0], step),
            indexing="xy"),
        axis=-1)\
        .astype(numpy.float32)
    p, q = p0.shape[:2]
    p0 = p0.reshape((p * q, 1, 2))
    p1 = cv2.calcOpticalFlowPyrLK(
        prev_grey,
        next_grey,
        p0,
        None,
        winSize=(win_size, win_size),
        maxLevel=max_level)[0]
    flow = p1.reshape((p, q, 2)) - p0.reshape((p, q, 2))
    if step == 1:
        return flow
    return numpy.kron(flow, numpy.ones((step, step, 1)))[0:m,0:n,:].astype(flow.dtype)


class CvFlowSource(FlowSource):

    def __init__(self, file: str, config: CvFlowConfig,
        size: tuple[int, int] | None = None,
        direction: FlowDirection | None = None, **src_args):
        FlowSource.__init__(self,
            direction if direction is not None else FlowDirection.FORWARD,
            **src_args)
        self.file = file
        self.config = config
        self.size = size
        self.capture = None
        self.prev_gray = None
        self.prev_rgb = None

    def setup(self):
        FlowSource.setup(self)
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
                resized = cv2.resize(frame, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
                self.prev_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                self.prev_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.prev_flow = None

    def next(self) -> numpy.ndarray:
        success, frame = self.capture.read()
        if frame is None or not success:
            raise StopIteration
        resized = cv2.resize(frame, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self.direction == FlowDirection.FORWARD:
            left_gray, right_gray = self.prev_gray, gray
            left_rgb, right_rgb = self.prev_rgb, rgb
        elif self.direction == FlowDirection.BACKWARD:
            left_gray, right_gray = gray, self.prev_gray
            left_rgb, right_rgb = rgb, self.prev_rgb
        else:
            raise ValueError(f"Invalid flow direction '{self.direction}'")
        if self.config.method == FlowMethod.FARNEBACK:
            flow = cv2.calcOpticalFlowFarneback(
                prev=left_gray,
                next=right_gray,
                flow=self.prev_flow.copy() if self.prev_flow is not None else None,
                pyr_scale=self.config.fb_pyr_scale,
                levels=self.config.fb_levels,
                winsize=self.config.fb_winsize,
                iterations=self.config.fb_iterations,
                poly_n=self.config.fb_poly_n,
                poly_sigma=self.config.fb_poly_sigma,
                flags=self.config.fb_flags,
            )
        elif self.config.method == FlowMethod.HORN_SCHUNCK:
            flow = calc_optical_flow_horn_schunck(
                prev_grey=left_gray,
                next_grey=right_gray,
                flow=self.prev_flow.copy() if self.prev_flow is not None else None,
                alpha=self.config.hs_alpha,
                max_iters=self.config.hs_iterations,
                decay=self.config.hs_decay,
                delta=self.config.hs_delta,
            )
        elif self.config.method == FlowMethod.LUKAS_KANADE:
            flow = calc_optical_flow_lukas_kanade(
                prev_grey=left_gray,
                next_grey=right_gray,
                win_size=self.config.lk_window_size,
                max_level=self.config.lk_max_level,
                step=self.config.lk_step
            )
        elif self.config.method == FlowMethod.LITEFLOWNET:
            try:
                from .liteflownet import calc_optical_flow_liteflownet
            except Exception as err:
                raise ImportError("LiteFlowNet method cannot be used. Are 'cupy' and 'torch' modules installed correctly?") from err
            flow = calc_optical_flow_liteflownet(left_rgb, right_rgb)
        self.prev_gray = gray
        self.prev_rgb = rgb
        return flow

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.capture.release()


class ArchiveFlowSource(FlowSource):

    def __init__(self, path: str, **src_args):
        FlowSource.__init__(self, FlowDirection.FORWARD, **src_args)
        self.path = path
        self.archive = None

    def setup(self):
        FlowSource.setup(self)
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
        return flow

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.archive.close()
