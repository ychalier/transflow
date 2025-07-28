import dataclasses
import os
import re
import sys
import time

from .flow import Direction, LockMode


@dataclasses.dataclass
class Config:

    # Positional Args
    flow_path: str
    bitmap_path: str | None
    output_path: str | list[str] | None
    extra_flow_paths: list[str] | None

    # Flow Args
    flows_merging_function: str = "first"
    use_mvs: bool = False
    mask_path: str | None = None
    kernel_path: str | None = None
    cv_config: str | None = None
    flow_filters: str | None = None
    direction: str | Direction = "forward"
    seek_time: float | None = None
    duration_time: float | None = None
    repeat: int = 1
    lock_expr: str | None = None
    lock_mode: str | LockMode = LockMode.STAY

    # Bitmap Args
    bitmap_seek_time: float | None = None
    bitmap_alteration_path: str | None = None
    bitmap_repeat: int = 1

    # Accumulator Args
    reset_mode: str = "off"
    reset_alpha: float = .9
    reset_mask_path: str | None = None
    heatmap_mode: str = "discrete"
    heatmap_args: str = "0:4:2:1"
    heatmap_reset_threshold: float | None = None
    acc_method: str = "map"
    accumulator_background: str = "ffffff"
    stack_composer: str = "top"
    initial_canvas: str | None = None
    bitmap_mask_path: str | None = None
    crumble: bool = False
    bitmap_introduction_flags: int = 1

    # Output Args
    vcodec: str = "h264"
    size: str | tuple[int, int] | None = None
    output_intensity: bool = False
    output_heatmap: bool = False
    output_accumulator: bool = False
    render_scale: float = 1
    render_colors: str | tuple[str, ...] | None = None
    render_binary: bool = False

    # General Args
    seed: int | None = None

    def todict(self) -> dict:
        now = time.time()
        d = dataclasses.asdict(self)
        d["timestamp"] = now
        if isinstance(self.direction, Direction):
            d["direction"] = self.direction.value
        if isinstance(self.lock_mode, LockMode):
            d["lock_mode"] = self.lock_mode.value
        d["command"] = {
            "executable": sys.executable,
            "argv": sys.argv
        }
        return d

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
