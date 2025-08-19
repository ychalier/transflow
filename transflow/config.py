import os
import random
import re
import sys
import time

from .flow import Direction, LockMode
from .utils import parse_timestamp


class PixmapSourceConfig:

    def __init__(self,
            path: str,
            seek_time: float | str | None = None,
            alteration_path: str | None = None,
            repeat: int | None = 1,
            layer: int | None = None):
        self.path: str = path
        self.seek_time: float | None = parse_timestamp(seek_time)
        self.alteration_path: str | None = alteration_path
        self.repeat: int = 1 if repeat is None else repeat
        self.layer: int = 0 if layer is None else layer

    @classmethod
    def fromdict(cls, d: dict):
        return cls(
            d["path"],
            seek_time=d.get("seek_time", None),
            alteration_path=d.get("alteration_path", None),
            repeat=d.get("repeat", 1),
            layer=d.get("layer", None),
        )

    def todict(self) -> dict:
        return {
            "path": self.path,
            "seek_time": self.seek_time,
            "alteration_path": self.alteration_path,
            "repeat": self.repeat,
            "layer": self.layer,
        }


def parse_bool_arg(arg: bool | str | None, default: bool) -> bool:
    if arg is None:
        return default
    if isinstance(arg, str):
        return arg.lower().strip() in ("1", "on", "o", "oui", "yes", "y")
    return arg


class LayerConfig:

    def __init__(self,
            index: int,
            classname: str | None = None,
            mask_src: str | None = None,
            mask_dst: str | None = None,
            mask_alpha: str | None = None,
            transparent_pixels_can_move: bool | str | None = None,
            pixels_can_move_to_empty_spot: bool | str | None = None,
            pixels_can_move_to_filled_spot: bool | str | None = None,
            moving_pixels_leave_empty_spot: bool | str | None = None,
            reset_mode: str | None = None,
            reset_mask: str | None = None,
            reset_pixels_leave_healed_spot: bool | str | None = None,
            reset_pixels_leave_empty_spot: bool | str | None = None,
            reset_random_factor: float | None = None,
            reset_constant_step: float | None = None,
            reset_linear_factor: float | None = None,
            mask_introduction: str | None = None,
            introduce_pixels_on_empty_spots: bool | None = None,
            introduce_pixels_on_filled_spots: bool | None = None,
            introduce_moving_pixels: bool | None = None,
            introduce_unmoving_pixels: bool | None = None,
            introduce_once: bool | None = None,
            ):
        self.index: int = index
        self.classname = "moveref" if classname is None else classname
        self.mask_src = mask_src
        self.mask_dst = mask_dst
        self.mask_alpha = mask_alpha
        self.transparent_pixels_can_move = parse_bool_arg(transparent_pixels_can_move, False)
        self.pixels_can_move_to_empty_spot = parse_bool_arg(pixels_can_move_to_empty_spot, True)
        self.pixels_can_move_to_filled_spot = parse_bool_arg(pixels_can_move_to_filled_spot, True)
        self.moving_pixels_leave_empty_spot = parse_bool_arg(moving_pixels_leave_empty_spot, False)
        self.reset_mode = "off" if reset_mode is None else reset_mode
        self.reset_mask = reset_mask
        self.reset_pixels_leave_healed_spot = parse_bool_arg(reset_pixels_leave_healed_spot, True)
        self.reset_pixels_leave_empty_spot = parse_bool_arg(reset_pixels_leave_empty_spot, True)
        self.reset_random_factor = 1 if reset_random_factor is None else reset_random_factor
        self.reset_constant_step = 1 if reset_constant_step is None else reset_constant_step
        self.reset_linear_factor = 0.1 if reset_linear_factor is None else reset_linear_factor
        self.mask_introduction = mask_introduction
        self.introduce_pixels_on_empty_spots = parse_bool_arg(introduce_pixels_on_empty_spots, True)
        self.introduce_pixels_on_filled_spots = parse_bool_arg(introduce_pixels_on_filled_spots, True)
        self.introduce_moving_pixels = parse_bool_arg(introduce_moving_pixels, True)
        self.introduce_unmoving_pixels = parse_bool_arg(introduce_unmoving_pixels, True)
        self.introduce_once = parse_bool_arg(introduce_once, False)

    @classmethod
    def fromdict(cls, d: dict):
        return cls(
            d["index"],
            classname=d.get("classname", "reference"),
            reset_mode=d.get("reset_mode", "off"),
            mask_src=d.get("mask_src", None),
            mask_dst=d.get("mask_dst", None),
            mask_alpha=d.get("mask_alpha", None),
            reset_mask=d.get("reset_mask", None),
            transparent_pixels_can_move=d.get("transparent_pixels_can_move", False),
            pixels_can_move_to_empty_spot=d.get("pixels_can_move_to_empty_spot", True),
            pixels_can_move_to_filled_spot=d.get("pixels_can_move_to_filled_spot", True),
            moving_pixels_leave_empty_spot=d.get("moving_pixels_leave_empty_spot", False),
            reset_pixels_leave_healed_spot=d.get("reset_pixels_leave_healed_spot", True),
            reset_pixels_leave_empty_spot=d.get("reset_pixels_leave_empty_spot", True),
            reset_random_factor=d.get("reset_random_factor", 1),
            reset_constant_step=d.get("reset_constant_step", 1),
            reset_linear_factor=d.get("reset_linear_factor", 0.1),
            mask_introduction=d.get("mask_introduction", None),
            introduce_pixels_on_empty_spots=d.get("introduce_pixels_on_empty_spots", True),
            introduce_pixels_on_filled_spots=d.get("introduce_pixels_on_filled_spots", True),
            introduce_moving_pixels=d.get("introduce_moving_pixels", True),
            introduce_unmoving_pixels=d.get("introduce_unmoving_pixels", True),
            introduce_once=d.get("introduce_once", False),
        )

    def todict(self) -> dict:
        return {
            "index": self.index,
            "classname": self.classname,
            "mask_src": self.mask_src,
            "mask_dst": self.mask_dst,
            "mask_alpha": self.mask_alpha,
            "transparent_pixels_can_move": self.transparent_pixels_can_move,
            "pixels_can_move_to_empty_spot": self.pixels_can_move_to_empty_spot,
            "pixels_can_move_to_filled_spot": self.pixels_can_move_to_filled_spot,
            "moving_pixels_leave_empty_spot": self.moving_pixels_leave_empty_spot,
            "reset_mode": self.reset_mode,
            "reset_mask": self.reset_mask,
            "reset_pixels_leave_healed_spot": self.reset_pixels_leave_healed_spot,
            "reset_pixels_leave_empty_spot": self.reset_pixels_leave_empty_spot,
            "reset_random_factor": self.reset_random_factor,
            "reset_constant_step": self.reset_constant_step,
            "reset_linear_factor": self.reset_linear_factor,
            "mask_introduction": self.mask_introduction,
            "introduce_pixels_on_empty_spots": self.introduce_pixels_on_empty_spots,
            "introduce_pixels_on_filled_spots": self.introduce_pixels_on_filled_spots,
            "introduce_moving_pixels": self.introduce_moving_pixels,
            "introduce_unmoving_pixels": self.introduce_unmoving_pixels,
            "introduce_once": self.introduce_once,
        }


class Config:

    def __init__(self,
            flow_path: str,
            extra_flow_paths: list[str] | None = None,
            flows_merging_function: str = "first",
            use_mvs: bool = False,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            cv_config: str | None = None,
            flow_filters: str | None = None,
            direction: str | int | Direction = "forward",
            seek_time: float | str | None = None,
            duration_time: float | str | None = None,
            to_time: float | str | None = None,
            repeat: int = 1,
            lock_expr: str | None = None,
            lock_mode: str | int | LockMode = LockMode.STAY,
            pixmap_sources: list[PixmapSourceConfig] = [],
            layers: list[LayerConfig] = [],
            compositor_background: str | None = None,
            output_path: str | list[str] | None = None,
            vcodec: str = "h264",
            size: str | tuple[int, int] | list[int] | None = None,
            output_intensity: bool = False,
            render_scale: float = 1,
            render_colors: str | tuple[str, ...] | list[str] | None = None,
            render_binary: bool = False,
            seed: int | None = None,
            ):

        # Flow Args
        self.flow_path: str = flow_path
        self.extra_flow_paths: list[str] = [] if extra_flow_paths is None else extra_flow_paths
        self.flows_merging_function: str = flows_merging_function
        if not self.extra_flow_paths:
            self.flows_merging_function = "first"
        self.use_mvs: bool = use_mvs
        self.mask_path: str | None = mask_path
        self.kernel_path: str | None = kernel_path
        self.cv_config: str | None = cv_config
        self.flow_filters: str | None = flow_filters
        self.direction: Direction = Direction.from_arg(direction)
        parsed_seek_time = parse_timestamp(seek_time)
        self.seek_time: float = 0 if parsed_seek_time is None else parsed_seek_time
        self.duration_time: float | None = None
        parsed_duration_time = parse_timestamp(duration_time)
        parsed_to_time = parse_timestamp(to_time)
        if parsed_to_time is not None:
            self.duration_time = parsed_to_time - self.seek_time
        else:
            self.duration_time = parsed_duration_time
        if self.duration_time is not None and self.duration_time < 0:
            raise ValueError(f"Duration must be positive (got {duration_time:f})")
        self.repeat: int = repeat
        self.lock_expr: str | None = lock_expr
        self.lock_mode: LockMode = LockMode.from_arg(lock_mode)

        # Pixmap Args
        self.pixmap_sources = pixmap_sources

        # Compositor Args
        self.layers = layers
        layer_indices = set()
        for layer in self.layers:
            if layer.index in self.layers:
                raise ValueError(f"Duplicate layer index {layer.index}")
            layer_indices.add(layer.index)
        for pixmap_config in self.pixmap_sources:
            if pixmap_config.layer not in layer_indices:
                self.layers.append(LayerConfig(pixmap_config.layer))
                layer_indices.add(pixmap_config.layer)
        self.compositor_background: str = "#FFFFFF" if compositor_background is None else compositor_background

        # Output Args
        self.output_path: str | list[str] | None = None if (isinstance(output_path, list) and not output_path) else output_path
        self.vcodec: str = vcodec
        if isinstance(size, str):
            size_split = re.split(r"[^\d]", size)[0]
            size = (int(size_split[0]), int(size_split[1]))
        if isinstance(size, list):
            size = (size[0], size[1])
        self.size: tuple[int, int] | None = size
        self.output_intensity: bool = output_intensity
        self.render_scale: float = render_scale
        if isinstance(render_colors, str):
            render_colors = tuple(render_colors.split(","))
        elif isinstance(render_colors, list):
            render_colors = tuple(render_colors)
        self.render_colors: tuple[str, ...] | None = render_colors
        self.render_binary: bool = render_binary

        # General Args
        self.seed: int = random.randint(0, 2**32-1) if seed is None else seed

    @classmethod
    def fromdict(cls, d: dict):
        return cls(
            d["flow_path"],
            extra_flow_paths=d.get("extra_flow_paths", None),
            flows_merging_function=d.get("flows_merging_function", "first"),
            use_mvs=d.get("use_mvs", False),
            mask_path=d.get("mask_path", None),
            kernel_path=d.get("kernel_path", None),
            cv_config=d.get("cv_config", None),
            flow_filters=d.get("flow_filters", None),
            direction=d.get("direction", "forward"),
            seek_time=d.get("seek_time", None),
            duration_time=d.get("duration_time", None),
            to_time=d.get("to_time", None),
            repeat=d.get("repeat", 1),
            lock_expr=d.get("lock_expr", None),
            lock_mode=d.get("lock_mode", LockMode.STAY),
            pixmap_sources=[PixmapSourceConfig.fromdict(dd) for dd in d.get("pixmap_sources", [])],
            layers=[LayerConfig.fromdict(dd) for dd in d.get("layers", [])],
            compositor_background=d.get("compositor_background", "#ffffff"),
            output_path=d.get("output_path", None),
            vcodec=d.get("vcodec", "h264"),
            size=d.get("size", None),
            output_intensity=d.get("output_intensity", False),
            render_scale=d.get("render_scale", 1),
            render_colors=d.get("render_colors", None),
            render_binary=d.get("render_binary", False),
            seed=d.get("seed", None),
        )

    def todict(self) -> dict:
        return {
            "flow_path": self.flow_path,
            "extra_flow_paths": self.extra_flow_paths,
            "flows_merging_function": self.flows_merging_function,
            "use_mvs": self.use_mvs,
            "mask_path": self.mask_path,
            "kernel_path": self.kernel_path,
            "cv_config": self.cv_config,
            "flow_filters": self.flow_filters,
            "direction": self.direction.value,
            "seek_time": self.seek_time,
            "duration_time": self.duration_time,
            "repeat": self.repeat,
            "lock_expr": self.lock_expr,
            "lock_mode": self.lock_mode.value,
            "pixmap_sources": [x.todict() for x in self.pixmap_sources],
            "layers": [x.todict() for x in self.layers],
            "compositor_background": self.compositor_background,
            "output_path": self.output_path,
            "vcodec": self.vcodec,
            "size": self.size,
            "output_intensity": self.output_intensity,
            "render_scale": self.render_scale,
            "render_colors": self.render_colors,
            "render_binary": self.render_binary,
            "seed": self.seed,
            "timestamp": time.time(),
            "command": {
                "executable": sys.executable,
                "argv": sys.argv
            },
        }

    def get_secondary_output_path(self, suffix: str) -> str:
        base_output_path = None
        if isinstance(self.output_path, list):
            mjpeg_pattern = re.compile(r"^mjpeg(:[:a-z0-9A-Z\-]+)?$", re.IGNORECASE)
            for path in self.output_path:
                if mjpeg_pattern.match(path):
                    continue
                base_output_path = path
                break
        else:
            base_output_path = self.output_path
        path = os.path.splitext(self.flow_path if base_output_path is None else base_output_path)[0]
        if path.endswith(".flow") or path.endswith(".ckpt"):
            path = path[:-5]
        if re.match(r".*\.(\d{3})$", path):
            path = path[:-4]
        return path + suffix
