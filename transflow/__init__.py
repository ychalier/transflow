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

    class Formatter(argparse.ArgumentDefaultsHelpFormatter):

        def _format_args(self, action, default_metavar):
            if action.nargs == "+" and isinstance(action.metavar, tuple) and len(action.metavar) == 2:
                m1, m2 = self._metavar_formatter(action, default_metavar)(2)
                if action.choices:
                    try:
                        choices = sorted(action.choices)
                    except TypeError:
                        # fall back if choices aren't sortable
                        choices = list(action.choices)
                    if isinstance(action, AppendLayer):
                        m2 = "{" + ",".join(map(str, choices)) + "}"
                    else:
                        m1 = "{" + ",".join(map(str, choices)) + "}"
                return f"{m1} [{m2}]"
            return super()._format_args(action, default_metavar)

    # class AWithOptionalB(argparse.Action):
    #     def __call__(self, parser, namespace, values, option_string=None):
    #         # enforce 1 or 2 values
    #         if not (1 <= len(values) <= 2):
    #             parser.error(f"{option_string} expects 1 or 2 arguments: A [B]")

    #         # first token: A with choices (use self.choices if provided)
    #         a = values[0]
    #         if self.choices is not None and a not in self.choices:
    #             parser.error(f"A must be one of {sorted(self.choices)} (got {a!r})")
    #         setattr(namespace, self.dest, a)  # store A under dest (e.g., 'a')

    #         # optional second token: B as float; if omitted, keep existing namespace.b
    #         if len(values) == 2:
    #             try:
    #                 b_val = float(values[1])
    #             except ValueError:
    #                 parser.error("B must be a floating-point number")
    #             setattr(namespace, "b", b_val)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=Formatter)
    parser.add_argument("-v", "--version", action="version", version=f"Transflow v{__version__}")

    # Flow Args
    group = parser.add_argument_group("flow options")
    group.add_argument("flow", type=str, help="input flow: a path to either a video file or a zip archive (precomputed flow or checkpoint)")
    group.add_argument("--flow", type=str, nargs="*", help="path to an additionnal flow source (video file or ZIP archive)")
    group.add_argument("--merge", type=str, default="sum", choices=["first", "sum", "average", "difference", "product", "maskbin", "masklin", "absmax"], help="operations to aggregate extra flow sources")
    group.add_argument("--mv", action="store_true", help="use motion vectors to compute the optical flow instead of OpenCV's Farneback algorithm")
    group.add_argument("--mask", type=str, default=None, help="path to an image to applay a per-pixel flow scale")
    group.add_argument("--kernel", type=str, default=None, help="path to a NPY file storing a convolution kernel to apply on the optical flow (impacts performances)")
    group.add_argument("-c", "--cv-config", type=str, default=None, help="path to a JSON file containing settings for OpenCV's Farneback algorithm or 'window' to use a live control window; if None, a default config is used")
    group.add_argument("-f", "--filters", type=str, default=None, help="list of flow filters, separated by semicolons; available filters are scale, threshold and clip; all take an expression as argument which can either be a constant or a Pythonic expression based on the variable `t`, the frame timestamp in seconds")
    group.add_argument("-d", "--direction", type=str, choices=["forward", "backward"], default="backward", help="direction of flow computation")
    group.add_argument("-s", "--seek", type=str, default=None, help="start timestamp for flow source")
    group.add_argument("-t", "--duration", type=str, default=None, help="max output duration")
    group.add_argument("--to", type=str, default=None, help="end timestamp for flow source")
    group.add_argument("--repeat", type=int, default=1, help="repeat flow inputs (0 to loop indefinitely)")
    # TODO: group into one argument like filters
    group.add_argument("--lock", type=str, default=None, help="Expression to lock the flow. In lock mode 'stay', a list of couples (start_t, duration). In lock mode 'skip', an expression based on variable `t`. Timings are the output frame timestamps, in seconds.")
    # group.add_argument("--lock-mode", type=str, default="stay", choices=["skip", "stay"], help="When the flow is locked, either pause the source ('stay') or continue reading it ('skip')")

    # Pixmap Args
    group = parser.add_argument_group("pixmap options")
    # TODO: same as reset param, optional second arg
    group.add_argument("-p", "--pixmap", action=AppendPixmap, nargs="+", metavar=("path", "layer"), type=str, help="input pixmap: either a path to a video or an image file or a still image generator (color, noise, bwnoise, cnoise, gradient, first); if None, the input flow will be preprocessed")
    # group.add_argument("-pl", "--pixmap-layer", action=SetForLastPixmapSource, type=int, default=0)
    group.add_argument("--alteration", action=SetPixmap, type=str, default=None, help="path to a PNG file containing alteration to apply to pixmap")
    group.add_argument("--pixmap-seek", action=SetPixmap, type=str, default=None, help="start timestamp for pixmap source")
    group.add_argument("--pixmap-repeat", action=SetPixmap, type=int, default=1, help="repeat pixmap input (0 to loop indefinitely)")

    # Compositor Args
    group = parser.add_argument_group("compositor options")
    group.add_argument("--background", type=str, default="#ffffff", help="compositor background color")

    # Layer Args
    group = parser.add_argument_group("layer options")
    group.add_argument("-l", "--layer", action=AppendLayer, nargs="+", metavar=("index", "class"), type=str, help="layer index")
    group.add_argument("--mask-alpha", dest="mask_alpha", action=SetLayer, type=str, default=None)
    group.add_argument("--move-mask-source", dest="mask_src", action=SetLayer, type=str, default=None)
    group.add_argument("--move-mask-destination", dest="mask_dst", action=SetLayer, type=str, default=None)
    group.add_argument("--move-from-empty", dest="transparent_pixels_can_move", action=ConstLayer, const=True)
    group.add_argument("--no-move-to-empty", dest="pixels_can_move_to_empty_spot", action=ConstLayer, const=False)
    group.add_argument("--no-move-to-filled", dest="pixels_can_move_to_filled_spot", action=ConstLayer, const=False)
    group.add_argument("-e", "--leave-empty-spot", dest="moving_pixels_leave_empty_spot", action=ConstLayer, const=True)
    group.add_argument("-r", "--reset", action=SetLayer, nargs="+", metavar=("mode", "factor"), type=str, choices=["off", "random", "constant", "linear"], default="off", help="layer reset mode")
    group.add_argument("--reset-mask", action=SetLayer, type=str)
    group.add_argument("--introduction-mask", action=SetLayer, type=str, default=None)
    group.add_argument("--no-introduce-on-empty", dest="introduce_pixels_on_empty_spots", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-on-filled", dest="introduce_pixels_on_filled_spots", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-moving", dest="introduce_moving_pixels", action=ConstLayer, const=False)
    group.add_argument("--no-introduce-unmoving", dest="introduce_unmoving_pixels", action=ConstLayer, const=False)
    group.add_argument("--introduce-once", dest="introduce_once", action=ConstLayer, const=True)
    group.add_argument("--introduce-on-all-filled", dest="introduce_on_all_filled_spots", action=ConstLayer, const=True)
    group.add_argument("--introduce-on-all-empty", dest="introduce_on_all_empty_spots", action=ConstLayer, const=True)

    # Output Args
    group = parser.add_argument_group("output options")
    group.add_argument("-o", "--output", type=str, action="append", help="output path: if provided, path to export the output video (as an MP4 file) ; otherwise, opens a temporary display window")
    group.add_argument("--vcodec", type=str, default="h264", help="video codec for the output video file")
    group.add_argument("--size", type=str, default=None, help="target video size, for webcams and generated pixmaps, of the form WIDTHxHEIGHT")
    group.add_argument("--view-flow", action="store_true", help="output flow intensity as a heatmap")
    group.add_argument("--render-scale", type=float, default=0.1, help="render scale for heatmap and accumulator output")
    group.add_argument("--render-colors", type=str, default=None, help="colors for rendering heatmap (2 colors) and accumulator (4 colors) outputs, hex format, separated by commas")
    group.add_argument("--render-binary", action="store_true", help="1d render will be binary, ie. no gradient will appear")

    # General Args
    group = parser.add_argument_group("general options")
    group.add_argument("--seed", type=int, default=None, help="random seed")

    # Pipeline Args
    group = parser.add_argument_group("processing options")
    group.add_argument("-S", "--safe", action="store_true", help="save a checkpoint when the program gets interrupted or an error occurs")
    group.add_argument("--checkpoint-every", type=int, default=None, help="export checkpoint every X frame")
    group.add_argument("--checkpoint-end", action="store_true", help="export checkpoint at the last frame")
    group.add_argument("--disable-exec", action="store_true", help="do not open the output video file when done")
    group.add_argument("--replace", action="store_true", help="overwrite any existing output file (by default, a new filename is generated to avoid conflicts and overwriting)")
    group.add_argument("--disable-config-export", action="store_true", help="disable automatic configuration export")
    group.add_argument("--enable-flow-export", action="store_true", help="export computed flow to a file")
    group.add_argument("--export-rounded-flow", action="store_true", help="export preprocessed flow as integer values (faster and lighter, but may introduce artefacts)")
    group.add_argument("-O", "--preview-output", action="store_true", help="preview output while exporting")
    group.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level", default="DEBUG")
    group.add_argument("--log-handler", type=str, help="Comma-separated list of log handlers (possible values are 'file', 'stream' or 'null' (default))", default="null")
    group.add_argument("--log-path", type=pathlib.Path, help="Path to log file (if 'file' log handler is used, see --log-handler)", default=pathlib.Path("transflow.log"))

    # GUI Args
    group = parser.add_argument_group("GUI options")
    group.add_argument("--gui-host", type=str, default="localhost", help="GUI host address")
    group.add_argument("--gui-port", type=int, default=8000, help="GUI port")
    group.add_argument("--gui-mjpeg-port", type=int, default=8001, help="GUI MJPEG port")

    args = parser.parse_args()
    print(args)

    return

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
            extra_flow_paths=args.extra_flow,
            flows_merging_function=args.flows_merging_function,
            use_mvs=args.use_mvs,
            mask_path=args.flow_mask,
            kernel_path=args.flow_kernel,
            cv_config=args.cv_config,
            flow_filters=args.flow_filters,
            direction=args.direction,
            seek_time=args.seek,
            duration_time=args.duration,
            to_time=args.to,
            repeat=args.repeat,
            lock_expr=args.lock,
            lock_mode=args.lock_mode,
            # Pixmap Args
            pixmap_sources=[
                PixmapSourceConfig(
                    d["path"],
                    seek_time=d.get("pixmap_seek"),
                    alteration_path=d.get("pixmap_alteration"),
                    repeat=d.get("pixmap_repeat"),
                    layer=d.get("pixmap_layer"),
                )
                for d in getattr(args, "pixmap_sources", [])
            ],
            # Compositor Args
            layers=[
                LayerConfig(
                    d["index"],
                    classname=d.get("layer_class"),
                    mask_src=d.get("layer_mask_src"),
                    mask_dst=d.get("layer_mask_dst"),
                    mask_alpha=d.get("layer_mask_alpha"),
                    transparent_pixels_can_move=d.get("layer_flag_movetransparent"),
                    pixels_can_move_to_empty_spot=d.get("layer_flag_movetoempty"),
                    pixels_can_move_to_filled_spot=d.get("layer_flag_movetofilled"),
                    moving_pixels_leave_empty_spot=d.get("layer_flag_leaveempty"),
                    reset_mode=d.get("layer_reset"),
                    reset_mask=d.get("layer_reset_mask"),
                    reset_random_factor=d.get("layer_reset_random_factor"),
                    reset_constant_step=d.get("layer_reset_constant_step"),
                    reset_linear_factor=d.get("layer_reset_linear_factor"),
                    mask_introduction=d.get("layer_mask_introduction"),
                    introduce_pixels_on_empty_spots=d.get("layer_introduce_empty"),
                    introduce_pixels_on_filled_spots=d.get("layer_introduce_filled"),
                    introduce_moving_pixels=d.get("layer_introduce_moving"),
                    introduce_unmoving_pixels=d.get("layer_introduce_unmoving"),
                    introduce_once=d.get("layer_introduce_once"),
                    introduce_on_all_filled_spots=d.get("layer_introduce_all_filled"),
                    introduce_on_all_empty_spots=d.get("layer_introduce_all_empty"),
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
        execute=not args.no_execute,
        replace=args.replace,
        export_config=not args.no_export_config,
        export_flow=args.export_flow,
        round_flow=args.round_flow,
        preview_output=args.preview_output,
        log_level=args.log_level,
        log_handler=args.log_handler,
        log_path=args.log_path,
    ).run()