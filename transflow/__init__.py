"""Transflow - Optical Flow Transfer

Set of tools for transferring optical flow from one media to another.
"""

__version__ = "1.10.0"
__author__ = "Yohan Chalier"
__license__ = "GNU GPLv3"
__maintainer__ = "Yohan Chalier"
__email__ = "yohan@chalier.fr"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__maintainer__",
    "__email__",
    "main",
]


def main():

    import argparse
    import pathlib

    class AppendPixmap(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            assert isinstance(values, list)
            elements = getattr(namespace, "pixmap_sources", None)
            if elements is None:
                elements = []
                setattr(namespace, "pixmap_sources", elements)
            if not values:
                parser.error("too few arguments for -p, --pixmap")
            if len(values) == 1:
                values.append(0)
            for i in range(1, len(values)):
                try:
                    values[i] = int(values[i])
                except ValueError:
                    parser.error(f"pixmap layer: invalid int value: '{values[i]}'")
            elements.append({"path": values[0], "layers": values[1:]})

    class SetPixmap(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            elements = getattr(namespace, "pixmap_sources", None)
            if not elements:
                parser.error(f"{option_string} must follow an -p/--pixmap")
            elements[-1][self.dest] = values

    class AppendLayer(argparse.Action):

        CLASSNAME_CHOICES = sorted(["moveref", "introduction", "static", "sum"])

        def __call__(self, parser, namespace, values, option_string=None):
            assert isinstance(values, list)
            elements = getattr(namespace, "layers", None)
            if elements is None:
                elements = []
                setattr(namespace, "layers", elements)
            if len(values) == 1:
                index, classname = values[0], "moveref"
            elif len(values) == 2:
                index, classname = values[0], values[1]
            else:
                parser.error("too many arguments for -l, --layer")
            try:
                index = int(index)
            except ValueError:
                parser.error(f"layer index: invalid int value: '{index}'")
            if not classname in self.CLASSNAME_CHOICES:
                choosefrom = ", ".join([f"'{cls}'" for cls in self.CLASSNAME_CHOICES])
                parser.error(f"layer class: invalid choice: '{classname}' (choose from {choosefrom})")
            elements.append({"index": index, "classname": classname})

    class SetLayer(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            elements = getattr(namespace, "layers", None)
            if not elements:
                elements = []
                setattr(namespace, "layers", elements)
                elements.append({"index", 0})
            elements[-1][self.dest] = values

    class ConstLayer(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            elements = getattr(namespace, "layers", None)
            if not elements:
                elements = []
                setattr(namespace, "layers", elements)
                elements.append({"index", 0})
            elements[-1][self.dest] = self.const
    
    class ResetAction(argparse.Action):

        RESET_CHOICES = sorted(["off", "random", "constant", "linear"])

        def __call__(self, parser, namespace, values, option_string=None):
            assert isinstance(values, list)
            if len(values) == 1:
                reset_mode, reset_factor = values[0], 1
            elif len(values) == 2:
                reset_mode, reset_factor = values
            else:
                parser.error("reset: invalid number of arguments, expected 1 or 2")
            if not reset_mode in self.RESET_CHOICES:
                choosefrom = ", ".join([f"'{cls}'" for cls in self.RESET_CHOICES])
                parser.error(f"reset mode: invalid choice: '{reset_mode}' (choose from {choosefrom})")
            try:
                reset_factor = float(reset_factor)
            except ValueError:
                parser.error(f"reset factor: invalid float value: '{reset_factor}'")
            elements = getattr(namespace, "layers", None)
            if not elements:
                elements = []
                setattr(namespace, "layers", elements)
                elements.append({"index", 0})
            elements[-1]["reset_mode"] = reset_mode
            elements[-1]["reset_factor"] = reset_factor
    
    class LockAction(argparse.Action):

        LOCKMODE_CHOICES = sorted(["stay", "skip"])

        def __call__(self, parser, namespace, values, option_string=None):
            assert isinstance(values, list)
            if not len(values) == 2:
                parser.error("lock: invalid number of arguments, expected 2")
            lock_mode, lock_expr = values
            if not lock_mode in self.LOCKMODE_CHOICES:
                choosefrom = ", ".join([f"'{cls}'" for cls in self.LOCKMODE_CHOICES])
                parser.error(f"lock mode: invalid choice: '{lock_mode}' (choose from {choosefrom})")
            setattr(namespace, "lock_mode", lock_mode)
            setattr(namespace, "lock_expr", lock_expr)

    class Formatter(argparse.ArgumentDefaultsHelpFormatter):

        def _format_args(self, action, default_metavar):
            if isinstance(action, AppendLayer):
                m1, m2 = self._metavar_formatter(action, default_metavar)(2)
                m2 = "{" + ",".join(map(str, AppendLayer.CLASSNAME_CHOICES)) + "}"
                return f"{m1} [{m2}]"
            elif isinstance(action, ResetAction):
                return "{" + ",".join(map(str, ResetAction.RESET_CHOICES)) + "} [RESET_FACTOR]"
            elif isinstance(action, LockAction):
                return "{" + ",".join(map(str, LockAction.LOCKMODE_CHOICES)) + "} LOCK_EXPR"
            return super()._format_args(action, default_metavar)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    parser.add_argument("-v", "--version", action="version", version=f"Transflow v{__version__}")

    # Flow Args
    group = parser.add_argument_group("flow options")
    group.add_argument("flow", type=str, help="input flow: a path to either a video file or a zip archive (precomputed flow or checkpoint)")
    group.add_argument("--flow", dest="extra_flow_paths", type=str, nargs="*", help="path to an additionnal flow source (video file or ZIP archive)")
    group.add_argument("--merge", dest="flows_merging_function", type=str, default="sum", choices=["first", "sum", "average", "difference", "product", "maskbin", "masklin", "absmax"], help="operations to aggregate extra flow sources")
    group.add_argument("--mv", dest="use_mvs", action="store_true", help="use motion vectors to compute the optical flow instead of OpenCV's Farneback algorithm")
    group.add_argument("--mask", dest="mask_path", type=str, default=None, help="path to an image to applay a per-pixel flow scale")
    group.add_argument("--kernel", dest="kernel_path", type=str, default=None, help="path to a NPY file storing a convolution kernel to apply on the optical flow (impacts performances)")
    group.add_argument("-c", "--cv-config", dest="cv_config", type=str, default=None, help="path to a JSON file containing settings for OpenCV's Farneback algorithm or 'window' to use a live control window; if None, a default config is used")
    group.add_argument("-f", "--filters", dest="flow_filters", type=str, default=None, help="list of flow filters, separated by semicolons; available filters are scale, threshold and clip; all take an expression as argument which can either be a constant or a Pythonic expression based on the variable `t`, the frame timestamp in seconds")
    group.add_argument("-d", "--direction", dest="direction", type=str, choices=["forward", "backward"], default="backward", help="direction of flow computation")
    group.add_argument("-s", "--seek", dest="seek_time", type=str, default=None, help="start timestamp for flow source")
    group.add_argument("-t", "--duration", dest="duration_time", type=str, default=None, help="max output duration")
    group.add_argument("--to", dest="to_time", type=str, default=None, help="end timestamp for flow source")
    group.add_argument("--repeat", dest="repeat", type=int, default=1, help="repeat flow inputs (0 to loop indefinitely)")
    group.add_argument("--lock", action=LockAction, nargs=2, type=str, help="Expression to lock the flow. In lock mode 'stay', a list of couples (start_t, duration). In lock mode 'skip', an expression based on variable `t`. Timings are the output frame timestamps, in seconds. When the flow is locked, either pause the source ('stay') or continue reading it ('skip')")

    # Pixmap Args
    group = parser.add_argument_group("pixmap options")
    group.add_argument("-p", "--pixmap", action=AppendPixmap, nargs="+", metavar=("path", "layer"), type=str, help="input pixmap: either a path to a video or an image file or a still image generator (color, noise, bwnoise, cnoise, gradient, first); if None, the input flow will be preprocessed")
    group.add_argument("--alteration", dest="pixmap_alteration", action=SetPixmap, type=str, default=None, help="path to a PNG file containing alteration to apply to pixmap")
    group.add_argument("--pixmap-seek", action=SetPixmap, type=str, default=None, help="start timestamp for pixmap source")
    group.add_argument("--pixmap-repeat", action=SetPixmap, type=int, default=1, help="repeat pixmap input (0 to loop indefinitely)")

    # Compositor Args
    group = parser.add_argument_group("compositor options")
    group.add_argument("--background", dest="compositor_background", type=str, default="#ffffff", help="compositor background color")

    # Layer Args
    group = parser.add_argument_group("layer options")
    group.add_argument("-l", "--layer", action=AppendLayer, nargs="+", metavar=("index", "class"), type=str, help="layer index", default="moveref")
    group.add_argument("--mask-alpha", dest="mask_alpha", action=SetLayer, type=str, default=None)
    group.add_argument("--move-mask-source", dest="mask_src", action=SetLayer, type=str, default=None)
    group.add_argument("--move-mask-destination", dest="mask_dst", action=SetLayer, type=str, default=None)
    group.add_argument("--move-from-empty", dest="transparent_pixels_can_move", action=ConstLayer, const=True)
    group.add_argument("--no-move-to-empty", dest="pixels_can_move_to_empty_spot", action=ConstLayer, const=False)
    group.add_argument("--no-move-to-filled", dest="pixels_can_move_to_filled_spot", action=ConstLayer, const=False)
    group.add_argument("-e", "--leave-empty-spot", dest="moving_pixels_leave_empty_spot", action=ConstLayer, const=True)
    group.add_argument("-r", "--reset", dest="reset", action=ResetAction, nargs="+", metavar=("mode", "factor"), type=str, default="off", help="layer reset mode")
    group.add_argument("--reset-mask", action=SetLayer, type=str)
    group.add_argument("--introduction-mask", dest="mask_introduction", action=SetLayer, type=str, default=None)
    group.add_argument("--no-introduce-on-empty", dest="introduce_pixels_on_empty_spots", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-on-filled", dest="introduce_pixels_on_filled_spots", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-moving", dest="introduce_moving_pixels", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-unmoving", dest="introduce_unmoving_pixels", action=ConstLayer, const=False)
    group.add_argument("-n", "--introduce-once", dest="introduce_once", action=ConstLayer, const=True)
    group.add_argument("-a", "--introduce-on-all-filled", dest="introduce_on_all_filled_spots", action=ConstLayer, const=True, nargs=0)
    group.add_argument("--introduce-on-all-empty", dest="introduce_on_all_empty_spots", action=ConstLayer, const=True)

    # Output Args
    group = parser.add_argument_group("output options")
    group.add_argument("-o", "--output", dest="output", type=str, action="append", help="output path: if provided, path to export the output video (as an MP4 file) ; otherwise, opens a temporary display window")
    group.add_argument("--vcodec", dest="vcodec", type=str, default="h264", help="video codec for the output video file")
    group.add_argument("--size", dest="size", type=str, default=None, help="target video size, for webcams and generated pixmaps, of the form WIDTHxHEIGHT")
    group.add_argument("--view-flow", dest="output_intensity", action="store_true", help="output flow intensity as a heatmap")
    group.add_argument("--render-scale", dest="render_scale", type=float, default=0.1, help="render scale for heatmap and accumulator output")
    group.add_argument("--render-colors", dest="render_colors", type=str, default=None, help="colors for rendering heatmap (2 colors) and accumulator (4 colors) outputs, hex format, separated by commas")
    group.add_argument("--render-binary", dest="render_binary", action="store_true", help="1d render will be binary, ie. no gradient will appear")

    # General Args
    group = parser.add_argument_group("general options")
    group.add_argument("--seed", dest="seed", type=int, default=None, help="random seed")

    # Pipeline Args
    group = parser.add_argument_group("processing options")
    group.add_argument("-S", "--safe", dest="safe", action="store_true", help="save a checkpoint when the program gets interrupted or an error occurs")
    group.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=None, help="export checkpoint every X frame")
    group.add_argument("--checkpoint-end", dest="checkpoint_end", action="store_true", help="export checkpoint at the last frame")
    group.add_argument("--no-exec", dest="execute", action="store_false", help="do not open the output video file when done")
    group.add_argument("--overwrite", dest="replace", action="store_true", help="overwrite any existing output file (by default, a new filename is generated to avoid conflicts and overwriting)")
    group.add_argument("--no-config-export", dest="export_config", action="store_false", help="disable automatic configuration export")
    group.add_argument("-F", "--export-flow", dest="export_flow", action="store_true", help="export computed flow to a file")
    group.add_argument("--export-rounded-flow", dest="round_flow", action="store_true", help="export preprocessed flow as integer values (faster and lighter, but may introduce artefacts)")
    group.add_argument("-O", "--preview-output", dest="preview_output", action="store_true", help="preview output while exporting")
    group.add_argument("--log-level", dest="log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level", default="DEBUG")
    group.add_argument("--log-handler", dest="log_handler", type=str, help="Comma-separated list of log handlers (possible values are 'file', 'stream' or 'null' (default))", default="null")
    group.add_argument("--log-path", dest="log_path", type=pathlib.Path, help="Path to log file (if 'file' log handler is used, see --log-handler)", default=pathlib.Path("transflow.log"))

    # GUI Args
    group = parser.add_argument_group("GUI options")
    group.add_argument("--gui-host", type=str, default="localhost", help="GUI host address")
    group.add_argument("--gui-port", type=int, default=8000, help="GUI port")
    group.add_argument("--gui-mjpeg-port", type=int, default=8001, help="GUI MJPEG port")

    args = parser.parse_args()

    if args.flow == "gui":
        from .gui import start_gui
        start_gui()
        return
    from .config import Config, PixmapSourceConfig, LayerConfig
    if args.flow.endswith(".json"):
        import json
        with open(args.flow, "r") as file:
            cfg = Config.fromdict(json.load(file))
    else:
        cfg = Config(
            # Flow Args
            args.flow,
            extra_flow_paths=args.extra_flow_paths,
            flows_merging_function=args.flows_merging_function,
            use_mvs=args.use_mvs,
            mask_path=args.mask_path,
            kernel_path=args.kernel_path,
            cv_config=args.cv_config,
            flow_filters=args.flow_filters,
            direction=args.direction,
            seek_time=args.seek_time,
            duration_time=args.duration_time,
            to_time=args.to_time,
            repeat=args.repeat,
            lock_expr=getattr(args, "lock_expr", None),
            lock_mode=getattr(args, "lock_mode", None),
            # Pixmap Args
            pixmap_sources=[
                PixmapSourceConfig(
                    d["path"],
                    seek_time=d.get("pixmap_seek"),
                    alteration_path=d.get("pixmap_alteration"),
                    repeat=d.get("pixmap_repeat"),
                    layers=d["layers"],
                )
                for d in getattr(args, "pixmap_sources", [])
            ],
            # Compositor Args
            layers=[
                LayerConfig(
                    d["index"],
                    classname=d.get("classname"),
                    mask_src=d.get("mask_src"),
                    mask_dst=d.get("mask_dst"),
                    mask_alpha=d.get("mask_alpha"),
                    transparent_pixels_can_move=d.get("transparent_pixels_can_move"),
                    pixels_can_move_to_empty_spot=d.get("pixels_can_move_to_empty_spot"),
                    pixels_can_move_to_filled_spot=d.get("pixels_can_move_to_filled_spot"),
                    moving_pixels_leave_empty_spot=d.get("moving_pixels_leave_empty_spot"),
                    reset_mode=d.get("reset_mode"),
                    reset_mask=d.get("reset_mask"),
                    reset_random_factor=d.get("reset_factor"),
                    reset_constant_step=d.get("reset_factor"),
                    reset_linear_factor=d.get("reset_factor"),
                    mask_introduction=d.get("mask_introduction"),
                    introduce_pixels_on_empty_spots=d.get("introduce_pixels_on_empty_spots"),
                    introduce_pixels_on_filled_spots=d.get("introduce_pixels_on_filled_spots"),
                    introduce_moving_pixels=d.get("introduce_moving_pixels"),
                    introduce_unmoving_pixels=d.get("introduce_unmoving_pixels"),
                    introduce_once=d.get("introduce_once"),
                    introduce_on_all_filled_spots=d.get("introduce_on_all_filled_spots"),
                    introduce_on_all_empty_spots=d.get("introduce_on_all_empty_spots"),
                )
                for d in getattr(args, "layers", [])
            ],
            compositor_background=args.compositor_background,
            # Output Args
            output_path=args.output,
            vcodec=args.vcodec,
            size=args.size,
            output_intensity=args.output_intensity,
            render_scale=args.render_scale,
            render_colors=args.render_colors,
            render_binary=args.render_binary,
            # General Args
            seed=args.seed,
        )
    from .pipeline import Pipeline
    Pipeline(cfg,
        safe=args.safe,
        checkpoint_every=args.checkpoint_every,
        checkpoint_end=args.checkpoint_end,
        execute=args.execute,
        replace=args.replace,
        export_config=args.export_config,
        export_flow=args.export_flow,
        round_flow=args.round_flow,
        preview_output=args.preview_output,
        log_level=args.log_level,
        log_handler=args.log_handler,
        log_path=args.log_path,
    ).run()