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

    class AppendAction(argparse.Action):

        LISTNAME = "foo"
        FIRSTARG = "bar"

        def __call__(self, parser, namespace, values, option_string=None):
            elements = getattr(namespace, self.LISTNAME, None)
            if elements is None:
                elements = []
                setattr(namespace, self.LISTNAME, elements)
            elements.append({self.FIRSTARG: values})


    class SetForLastAction(argparse.Action):

        LISTNAME = "foo"

        def __call__(self, parser, namespace, values, option_string=None):
            elements = getattr(namespace, self.LISTNAME, None)
            if not elements:
                parser.error(f"{option_string} does not have a {self.LISTNAME} to attach to")
            elements[-1][self.dest] = values


    class AppendPixmapSource(AppendAction):
        LISTNAME = "pixmap_sources"
        FIRSTARG = "path"


    class SetForLastPixmapSource(SetForLastAction):
        LISTNAME = "pixmap_sources"


    class AppendLayer(AppendAction):
        LISTNAME = "layers"
        FIRSTARG = "index"


    class SetForLastLayer(SetForLastAction):
        LISTNAME = "layers"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"Transflow v{__version__}")

    # Flow Args
    group = parser.add_argument_group("flow options")
    group.add_argument("flow", type=str, help="input flow: a path to either a video file or a zip archive (precomputed flow or checkpoint)")
    group.add_argument("--flow", dest="extra_flow", type=str, nargs="*", help="path to an additionnal flow source (video file or ZIP archive)")
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
    group.add_argument("-p", "--pixmap", action=AppendPixmapSource, type=str, help="input pixmap: either a path to a video or an image file or a still image generator (color, noise, bwnoise, cnoise, gradient, first); if None, the input flow will be preprocessed")
    group.add_argument("-pl", "--pixmap-layer", action=SetForLastPixmapSource, type=int, default=0)
    group.add_argument("-ps", "--pixmap-seek", action=SetForLastPixmapSource, type=str, default=None, help="start timestamp for pixmap source")
    group.add_argument("-pa", "--pixmap-alteration", action=SetForLastPixmapSource, type=str, default=None, help="path to a PNG file containing alteration to apply to pixmap")
    group.add_argument("-pr", "--pixmap-repeat", action=SetForLastPixmapSource, type=int, default=1, help="repeat pixmap input (0 to loop indefinitely)")

    # Compositor Args
    group = parser.add_argument_group("compositor options")
    group.add_argument("--background", type=str, default="#ffffff", help="compositor background color")

    # Layer Args
    # TODO: make it so, when no -l is specified, commands refer to layer 0
    group = parser.add_argument_group("layer options")
    group.add_argument("-l", "--layer", action=AppendLayer, type=int, help="layer index")
    group.add_argument("-lc", "--layer-class", action=SetForLastLayer, type=str, choices=["moveref", "introduction", "static", "sum"], default="moveref", help="layer class")
    group.add_argument("--mask-alpha", action=SetForLastLayer, type=str, default=None)
    group.add_argument("--mask-source", action=SetForLastLayer, type=str, default=None)
    group.add_argument("--mask-destination", action=SetForLastLayer, type=str, default=None)
    group.add_argument("--move-empty", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    group.add_argument("--move-to-empty", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("--move-to-filled", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("-E", "--leave-empty-spot", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    # TODO: merge into one reset param, interpreted in multiple ways
    group.add_argument("-r", "--reset", action=SetForLastLayer, type=str, choices=["off", "random", "constant", "linear"], default="off", help="layer reset mode")
    group.add_argument("-R", "--reset-param", action=SetForLastLayer, type=float, default=None)
    # group.add_argument("-lrr", "--layer-reset-random-factor", action=SetForLastLayer, type=float, default=0.1)
    # group.add_argument("-lrc", "--layer-reset-constant-step", action=SetForLastLayer, type=float, default=1)
    # group.add_argument("-lrl", "--layer-reset-linear-factor", action=SetForLastLayer, type=float, default=0.1)
    group.add_argument("--reset-mask", action=SetForLastLayer, type=str)
    group.add_argument("--introduction-mask", action=SetForLastLayer, type=str, default=None)
    group.add_argument("--introduce-on-empty", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("--introduce-on-filled", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("--introduce-moving", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("--introduce-unmoving", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    group.add_argument("--introduce-once", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    group.add_argument("--introduce-on-all-filled", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    group.add_argument("--introduce-on-all-empty", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    

    # Output Args
    group = parser.add_argument_group("output options")
    group.add_argument("-o", "--output", type=str, action="append", help="output path: if provided, path to export the output video (as an MP4 file) ; otherwise, opens a temporary display window")
    group.add_argument("--vcodec", type=str, default="h264", help="video codec for the output video file")
    group.add_argument("--size", type=str, default=None, help="target video size, for webcams and generated pixmaps, of the form WIDTHxHEIGHT")
    group.add_argument("--intensity", action="store_true", help="output flow intensity as a heatmap")
    group.add_argument("--render-scale", type=float, default=0.1, help="render scale for heatmap and accumulator output")
    group.add_argument("--render-colors", type=str, default=None, help="colors for rendering heatmap (2 colors) and accumulator (4 colors) outputs, hex format, separated by commas")
    group.add_argument("--render-binary", action="store_true", help="1d render will be binary, ie. no gradient will appear")

    # General Args
    group = parser.add_argument_group("general options")
    group.add_argument("--seed", type=int, default=None, help="random seed")

    # Pipeline Args
    group = parser.add_argument_group("processing options")
    group.add_argument("-s", "--safe", action="store_true", help="save a checkpoint when the program gets interrupted or an error occurs")
    group.add_argument("-ce", "--checkpoint-every", type=int, default=None, help="export checkpoint every X frame")
    group.add_argument("-cd", "--checkpoint-end", action="store_true", help="export checkpoint at the last frame")
    group.add_argument("-nx", "--no-execute", action="store_true", help="do not open the output video file when done")
    group.add_argument("-re", "--replace", action="store_true", help="overwrite any existing output file (by default, a new filename is generated to avoid conflicts and overwriting)")
    group.add_argument("-nc", "--no-export-config", action="store_true", help="disable automatic configuration export")
    group.add_argument("-ef", "--export-flow", action="store_true", help="export computed flow to a file")
    group.add_argument("-rf", "--round-flow", action="store_true", help="export preprocessed flow as integer values (faster and lighter, but may introduce artefacts)")
    group.add_argument("-po", "--preview-output", action="store_true", help="preview output while exporting")
    group.add_argument("-ll", "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level", default="DEBUG")
    group.add_argument("-lh", "--log-handler", type=str, help="Comma-separated list of log handlers (possible values are 'file', 'stream' or 'null' (default))", default="null")
    group.add_argument("-lp", "--log-path", type=pathlib.Path, help="Path to log file (if 'file' log handler is used, see --log-handler)", default=pathlib.Path("transflow.log"))

    # GUI Args
    group = parser.add_argument_group("GUI options")
    group.add_argument("-gh", "--gui-host", type=str, default="localhost", help="GUI host address")
    group.add_argument("-gp", "--gui-port", type=int, default=8000, help="GUI port")
    group.add_argument("-gm", "--gui-mjpeg-port", type=int, default=8001, help="GUI MJPEG port")

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