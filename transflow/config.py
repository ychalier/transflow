import os
import random
import re
import sys
import time

from .flow import Direction, LockMode
from .utils import parse_timestamp


class Config:

    def __init__(self,
            flow_path: str,
            bitmap_path: str | None,
            output_path: str | list[str] | None,
            extra_flow_paths: list[str] | None,
            flows_merging_function: str = "first",
            use_mvs: bool = False,
            mask_path: str | None = None,
            kernel_path: str | None = None,
            cv_config: str | None = None,
            flow_filters: str | None = None,
            direction: str | Direction = "forward",
            seek_time: float | str | None = None,
            duration_time: float | str | None = None,
            to_time: float | str | None = None,
            repeat: int = 1,
            lock_expr: str | None = None,
            lock_mode: str | LockMode = LockMode.STAY,
            bitmap_seek_time: float | str | None = None,
            bitmap_alteration_path: str | None = None,
            bitmap_repeat: int = 1,
            reset_mode: str = "off",
            reset_alpha: float = .9,
            reset_mask_path: str | None = None,
            heatmap_mode: str = "discrete",
            heatmap_args: str = "0:4:2:1",
            heatmap_reset_threshold: float | None = None,
            acc_method: str = "map",
            accumulator_background: str = "ffffff",
            stack_composer: str = "top",
            initial_canvas: str | None = None,
            bitmap_mask_path: str | None = None,
            crumble: bool = False,
            bitmap_introduction_flags: int = 1,
            vcodec: str = "h264",
            size: str | tuple[int, int] | None = None,
            output_intensity: bool = False,
            output_heatmap: bool = False,
            output_accumulator: bool = False,
            render_scale: float = 1,
            render_colors: str | tuple[str, ...] | None = None,
            render_binary: bool = False,
            seed: int | None = None,
            ):
        
        # Positional Args
        self.flow_path: str = flow_path
        self.bitmap_path: str | None = bitmap_path
        self.output_path: str | list[str] | None = None if (isinstance(output_path, list) and not output_path) else output_path
        self.extra_flow_paths: list[str] = [] if extra_flow_paths is None else extra_flow_paths
        
        # Flow Args
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
        
        # Bitmap Args
        self.bitmap_seek_time: float | None = parse_timestamp(bitmap_seek_time)
        self.bitmap_alteration_path: str | None = bitmap_alteration_path
        self.bitmap_repeat: int = bitmap_repeat
        
        # Accumulator Args
        self.reset_mode: str = reset_mode
        self.reset_alpha: float = reset_alpha
        self.reset_mask_path: str | None = reset_mask_path
        self.heatmap_mode: str = heatmap_mode
        self.heatmap_args: str = heatmap_args
        self.heatmap_reset_threshold: float | None = heatmap_reset_threshold
        self.acc_method: str = acc_method
        self.accumulator_background: str = accumulator_background
        self.stack_composer: str = stack_composer
        self.initial_canvas: str | None = initial_canvas
        self.bitmap_mask_path: str | None = bitmap_mask_path
        self.crumble: bool = crumble
        self.bitmap_introduction_flags: int = bitmap_introduction_flags
        
        # Output Args
        self.vcodec: str = vcodec
        if isinstance(size, str):
            size_split = re.split(r"[^\d]", size)[0]
            size = (int(size_split[0]), int(size_split[1]))
        self.size: tuple[int, int] | None = size
        self.output_intensity: bool = output_intensity
        self.output_heatmap: bool = output_heatmap
        self.output_accumulator: bool = output_accumulator
        self.render_scale: float = render_scale
        self.render_colors: tuple[str, ...] | None = tuple(render_colors.split(",")) if isinstance(render_colors, str) else render_colors
        self.render_binary: bool = render_binary
        
        # General Args
        self.seed: int = random.randint(0, 2**32-1) if seed is None else seed

    def todict(self) -> dict:
        return {
            "timestamp": time.time(),
            "command": {
                "executable": sys.executable,
                "argv": sys.argv
            },
            "flow_path": self.flow_path,
            "bitmap_path": self.bitmap_path,
            "output_path": self.output_path,
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
            "bitmap_seek_time": self.bitmap_seek_time,
            "bitmap_alteration_path": self.bitmap_alteration_path,
            "bitmap_repeat": self.bitmap_repeat,
            "reset_mode": self.reset_mode,
            "reset_alpha": self.reset_alpha,
            "reset_mask_path": self.reset_mask_path,
            "heatmap_mode": self.heatmap_mode,
            "heatmap_args": self.heatmap_args,
            "heatmap_reset_threshold": self.heatmap_reset_threshold,
            "acc_method": self.acc_method,
            "accumulator_background": self.accumulator_background,
            "stack_composer": self.stack_composer,
            "initial_canvas": self.initial_canvas,
            "bitmap_mask_path": self.bitmap_mask_path,
            "crumble": self.crumble,
            "bitmap_introduction_flags": self.bitmap_introduction_flags,
            "vcodec": self.vcodec,
            "size": self.size,
            "output_intensity": self.output_intensity,
            "output_heatmap": self.output_heatmap,
            "output_accumulator": self.output_accumulator,
            "render_scale": self.render_scale,
            "render_colors": self.render_colors,
            "render_binary": self.render_binary,
            "seed": self.seed,
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
