import enum
import json
import os
import re
import threading

import cv2
import numpy

from .source import FlowSource
from ..methods.horn_schunck import calc_optical_flow_horn_schunck
from ..methods.lukas_kanade import calc_optical_flow_lukas_kanade


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
        method = CvFlowSource.FlowMethod.from_string(method_name)
        self.config.update("method", method)
        for widget_name, widget in self.widgets.items():
            widget.setVisible(method_name == widget_name)
        for name, widget in self.inputs.items():
            self.config.update(name, widget.value())


class CvFlowConfig:

    def __init__(self,
            method: "str | CvFlowSource.FlowMethod" = "farneback",
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
        self.method = CvFlowSource.FlowMethod.from_string(method) if isinstance(method, str) else method
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
        self.window: CvFlowConfigWindow | None = None

    def start(self):
        if not self.show_window:
            return
        self.window = CvFlowConfigWindow(self)
        self.window.start()

    def update(self, attrname, value):
        if attrname == "method" and isinstance(value, str):
            value = CvFlowSource.FlowMethod.from_string(value)
        self.__setattr__(attrname, value)

    def reset(self):
        self.method = CvFlowSource.FlowMethod.FARNEBACK
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
            "method": CvFlowSource.FlowMethod.to_string(self.method),
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


class CvFlowSource(FlowSource):

    @enum.unique
    class FlowMethod(enum.Enum):
        FARNEBACK = 0
        HORN_SCHUNCK = 1
        LUKAS_KANADE = 2
        LITEFLOWNET = 3

        @classmethod
        def from_string(cls, string: str):
            if string == "farneback":
                return CvFlowSource.FlowMethod.FARNEBACK
            if string == "horn-schunck":
                return CvFlowSource.FlowMethod.HORN_SCHUNCK
            if string == "lukas-kanade":
                return CvFlowSource.FlowMethod.LUKAS_KANADE
            if string == "liteflownet":
                return CvFlowSource.FlowMethod.LITEFLOWNET
            raise ValueError(f"Invalid Flow Method: {string}")

        @staticmethod
        def to_string(method: "CvFlowSource.FlowMethod"):
            if method == CvFlowSource.FlowMethod.FARNEBACK:
                return "farneback"
            if method == CvFlowSource.FlowMethod.HORN_SCHUNCK:
                return "horn-schunck"
            if method == CvFlowSource.FlowMethod.LUKAS_KANADE:
                return "lukas-kanade"
            if method == CvFlowSource.FlowMethod.LITEFLOWNET:
                return "liteflownet"
            raise ValueError(f"Unknown flow method {method}")

    class Builder(FlowSource.Builder):

        def __init__(self,
                file: str,
                config: CvFlowConfig,
                size: tuple[int, int] | None = None,
                **kwargs):
            super().__init__(**kwargs)
            self.file = file
            self.config = config
            self.size = size
            self.capture: cv2.VideoCapture | None = None

        @property
        def cls(self):
            return CvFlowSource

        def build(self):
            if re.match(r"\d+", self.file):
                # file argument is probably a webcam index
                self.capture = cv2.VideoCapture(int(self.file))
            else:
                self.capture = cv2.VideoCapture(self.file)
            if self.size is not None:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.framerate = float(self.capture.get(cv2.CAP_PROP_FPS))
            self.base_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            super().build()
        
        def args(self):
            return [self.capture, self.config, *FlowSource.Builder.args(self)]

    def __init__(self, capture: cv2.VideoCapture, config: CvFlowConfig, *args, **kwargs):
        self.config = config
        self.capture = capture
        self.prev_gray = None
        self.prev_rgb = None
        self.config.start()
        FlowSource.__init__(self, *args, **kwargs)

    def rewind(self):
        self.input_frame_index = self.start_frame
        if self.capture is None:
            raise ValueError("Capture not initialized")
        if self.width is None or self.height is None:
            raise ValueError("Width or height not initialized")
        self.capture.set(cv2.CAP_PROP_POS_MSEC, 0)
        for i in range(self.input_frame_index + 1):
            success, frame = self.capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not open video at")
            if i == self.input_frame_index:
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
        if self.direction == FlowSource.FlowDirection.FORWARD:
            left_gray, right_gray = self.prev_gray, gray
            left_rgb, right_rgb = self.prev_rgb, rgb
        elif self.direction == FlowSource.FlowDirection.BACKWARD:
            left_gray, right_gray = gray, self.prev_gray
            left_rgb, right_rgb = rgb, self.prev_rgb
        else:
            raise ValueError(f"Invalid flow direction '{self.direction}'")
        if left_gray is None or right_gray is None:
            raise ValueError("Missing reference frames")
        if self.config.method == CvFlowSource.FlowMethod.FARNEBACK:
            flow = self.prev_flow.copy() if self.prev_flow is not None else numpy.zeros((self.height, self.width, 2), dtype=numpy.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev=left_gray,
                next=right_gray,
                flow=flow,
                pyr_scale=self.config.fb_pyr_scale,
                levels=self.config.fb_levels,
                winsize=self.config.fb_winsize,
                iterations=self.config.fb_iterations,
                poly_n=self.config.fb_poly_n,
                poly_sigma=self.config.fb_poly_sigma,
                flags=self.config.fb_flags,
            ).astype(numpy.float32)
        elif self.config.method == CvFlowSource.FlowMethod.HORN_SCHUNCK:
            flow = calc_optical_flow_horn_schunck(
                prev_grey=left_gray,
                next_grey=right_gray,
                flow=self.prev_flow.copy() if self.prev_flow is not None else None,
                alpha=self.config.hs_alpha,
                max_iters=self.config.hs_iterations,
                decay=self.config.hs_decay,
                delta=self.config.hs_delta,
            )
        elif self.config.method == CvFlowSource.FlowMethod.LUKAS_KANADE:
            flow = calc_optical_flow_lukas_kanade(
                prev_grey=left_gray,
                next_grey=right_gray,
                win_size=self.config.lk_window_size,
                max_level=self.config.lk_max_level,
                step=self.config.lk_step
            )
        elif self.config.method == CvFlowSource.FlowMethod.LITEFLOWNET:
            try:
                from ..methods.liteflownet import calc_optical_flow_liteflownet
            except Exception as err:
                raise ImportError("LiteFlowNet method cannot be used. Are 'cupy' and 'torch' modules installed correctly?") from err
            if left_rgb is None or right_rgb is None:
                raise ValueError("Missing reference frames")
            flow = calc_optical_flow_liteflownet(left_rgb, right_rgb)
        else:
            raise ValueError(f"Unknown flow method '{self.config.method}'")
        self.prev_gray = gray
        self.prev_rgb = rgb
        return flow

    def close(self):
        self.capture.release()
