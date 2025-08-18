"""Set of tools for transferring optical flow from one media to another.
"""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Flow Args
    parser.add_argument("flow", type=str, help="input flow: a path to either a video file or a zip archive (precomputed flow or checkpoint)")
    parser.add_argument("-f", "--extra-flow", type=str, nargs="*", help="merge multiple flow sources")
    parser.add_argument("-sm", "--flows-merging-function", type=str, default="sum", choices=["first", "sum", "average", "difference", "product", "maskbin", "masklin", "absmax"], help="operations to aggregate extra flow sources")
    parser.add_argument("-mv", "--use-mvs", action="store_true", help="use motion vectors to compute the optical flow instead of OpenCV's Farneback algorithm")
    parser.add_argument("-fm", "--flow-mask", type=str, default=None, help="path to an image to applay a per-pixel flow scale")
    parser.add_argument("-fk", "--flow-kernel", type=str, default=None, help="path to a NPY file storing a convolution kernel to apply on the optical flow (impacts performances)")
    parser.add_argument("-cc", "--cv-config", type=str, default=None, help="path to a JSON file containing settings for OpenCV's Farneback algorithm or 'window' to use a live control window; if None, a default config is used")
    parser.add_argument("-ff", "--flow-filters", type=str, default=None, help="list of flow filters, separated by semicolons; available filters are scale, threshold and clip; all take an expression as argument which can either be a constant or a Pythonic expression based on the variable `t`, the frame timestamp in seconds")
    parser.add_argument("-d", "--direction", type=str, choices=["forward", "backward"], default="backward", help="direction of flow computation")
    parser.add_argument("-ss", "--seek", type=str, default=None, help="start timestamp for flow source")
    parser.add_argument("-t", "--duration", type=str, default=None, help="max output duration")
    parser.add_argument("-to", "--to", type=str, default=None, help="end timestamp for flow source")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="repeat flow inputs (0 to loop indefinitely)")
    parser.add_argument("-lo", "--lock", type=str, default=None, help="Expression to lock the flow. In lock mode 'stay', a list of couples (start_t, duration). In lock mode 'skip', an expression based on variable `t`. Timings are the output frame timestamps, in seconds.")
    parser.add_argument("-lm", "--lock-mode", type=str, default="stay", choices=["skip", "stay"], help="When the flow is locked, either pause the source ('stay') or continue reading it ('skip')")

    # Bitmap Args
    parser.add_argument("-p", "--pixmap", action=AppendPixmapSource, type=str, help="input pixmap: either a path to a video or an image file or a still image generator (color, noise, bwnoise, cnoise, gradient, first); if None, the input flow will be preprocessed")
    parser.add_argument("-pl", "--pixmap-layer", action=SetForLastPixmapSource, type=int, default=0)
    # parser.add_argument("-lc", "--layer-class", action=SetForLastPixmapSource, type=str, choices=["reference", "introduction"], default="reference")
    parser.add_argument("-ps", "--pixmap-seek", action=SetForLastPixmapSource, type=str, default=None, help="start timestamp for pixmap source")
    parser.add_argument("-pa", "--pixmap-alteration", action=SetForLastPixmapSource, type=str, default=None, help="path to a PNG file containing alteration to apply to pixmap")
    parser.add_argument("-pr", "--pixmap-repeat", action=SetForLastPixmapSource, type=int, default=1, help="repeat pixmap input (0 to loop indefinitely)")
    
    # Compositor Args
    parser.add_argument("-l", "--layer", action=AppendLayer, type=int, help="layer index")
    parser.add_argument("-lc", "--layer-class", action=SetForLastLayer, type=str, choices=["reference", "introduction", "static"], default="reference", help="layer class")
    parser.add_argument("-lms", "--layer-mask-src", action=SetForLastLayer, type=str, default=None)
    parser.add_argument("-lmd", "--layer-mask-dst", action=SetForLastLayer, type=str, default=None)
    parser.add_argument("-lma", "--layer-mask-alpha", action=SetForLastLayer, type=str, default=None)
    parser.add_argument("-lmi", "--layer-mask-introduction", action=SetForLastLayer, type=str, default=None)
    parser.add_argument("-lft", "--layer-flag-movetransparent", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    parser.add_argument("-lfe", "--layer-flag-movetoempty", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-lff", "--layer-flag-movetofilled", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-lfl", "--layer-flag-leaveempty", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    parser.add_argument("-lr", "--layer-reset", action=SetForLastLayer, type=str, choices=["off", "random", "constant", "linear"], default="off", help="layer reset mode")
    parser.add_argument("-lmr", "--layer-mask-reset", action=SetForLastLayer, type=str, default=None)    
    parser.add_argument("-lfh", "--layer-flag-resetleavehealed", action=SetForLastLayer, type=str, default="on", choices=["on", "off"]) # TODO: consider removing those
    parser.add_argument("-lfr", "--layer-flag-resetleaveempty", action=SetForLastLayer, type=str, default="on", choices=["on", "off"]) # TODO: consider removing those
    parser.add_argument("-lrr", "--layer-reset-random-factor", action=SetForLastLayer, type=float, default=0.1)
    parser.add_argument("-lrc", "--layer-reset-constant-step", action=SetForLastLayer, type=float, default=1)
    parser.add_argument("-lrl", "--layer-reset-linear-factor", action=SetForLastLayer, type=float, default=0.1)
    parser.add_argument("-lie", "--layer-introduce-empty", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-lif", "--layer-introduce-filled", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-lim", "--layer-introduce-moving", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-liu", "--layer-introduce-unmoving", action=SetForLastLayer, type=str, default="on", choices=["on", "off"])
    parser.add_argument("-lio", "--layer-introduce-once", action=SetForLastLayer, type=str, default="off", choices=["on", "off"])
    
    # Accumulator Args
    # parser.add_argument("-m", "--acc-method", type=str, default="map", choices=["map", "stack", "sum", "crumble", "canvas"], help="accumulator method ('map' is default, 'stack' is very slow, 'sum' only works with backward flows)")
    # parser.add_argument("-rm", "--reset-mode", type=str, choices=["off", "random", "linear"], default="off", help="indices re-introduction mode")
    # parser.add_argument("-ra", "--reset-alpha", type=float, default=.1, help="indices re-introduction coefficient (effect my vary depending on mode)")
    # parser.add_argument("-rk", "--reset-mask", type=str, default=None, help="path to an image file to use a per-pixel re-introduction coefficient")
    # parser.add_argument("-hm", "--heatmap-mode", type=str, default="discrete", choices=["discrete", "continuous"], help="heatmap mode, discrete means binary (motion or not) and continuous means flow magnitude")
    # parser.add_argument("-ha", "--heatmap-args", type=str, default="0:4:2:1", help="in discrete mode, args are min:max:add:sub (ints), in continuous mode, args are max:decay:treshold (floats)")
    # parser.add_argument("-hr", "--heatmap-reset-threshold", type=float, default=None, help="heatmap threshold for reset effects")
    # parser.add_argument("-ab", "--accumulator-background", type=str, default="ffffff", help="background color used in stack and crumble remapper")
    # parser.add_argument("-sc", "--stack-composer", type=str, choices=["top", "add", "sub", "avg"], default="top", help="stack remapper compose function")
    # parser.add_argument("-ic", "--initial-canvas", type=str, default=None, help="set initial canvas for canvas accumulator, either a HEX color or a path to an image file")
    # parser.add_argument("-bm", "--bitmap-mask", type=str, default=None, help="path to a bitmap mask (black & white image) for canvas accumulator")
    # parser.add_argument("-cr", "--crumble", action="store_true", help="enable crumble effect for the canvas accumulator")
    # parser.add_argument("-bi", "--bitmap-introduction-flags", type=int, default=1, help="bitmap introduction flags for canvas accumulator (1 for motion, 2 for static, 3 for both)")

    # Output Args
    parser.add_argument("-o", "--output", type=str, action="append", help="output path: if provided, path to export the output video (as an MP4 file) ; otherwise, opens a temporary display window")
    parser.add_argument("-vc", "--vcodec", type=str, default="h264", help="video codec for the output video file")
    parser.add_argument("-sz", "--size", type=str, default=None, help="target video size, for webcams and generated bitmaps, of the form WIDTHxHEIGHT")
    parser.add_argument("-oi", "--output-intensity", action="store_true", help="output flow intensity as a heatmap")
    parser.add_argument("-oh", "--output-heatmap", action="store_true", help="output heatmap instead of transformed bitmap")
    parser.add_argument("-oa", "--output-accumulator", action="store_true", help="output accumulator instead of transformed bitmap")
    parser.add_argument("-rs", "--render-scale", type=float, default=0.1, help="render scale for heatmap and accumulator output")
    parser.add_argument("-rc", "--render-colors", type=str, default=None, help="colors for rendering heatmap (2 colors) and accumulator (4 colors) outputs, hex format, separated by commas")
    parser.add_argument("-rb", "--render-binary", action="store_true", help="1d render will be binary, ie. no gradient will appear")

    # General Args
    parser.add_argument("-sd", "--seed", type=int, default=None, help="random seed for generating bitmaps")

    # Pipeline Args
    parser.add_argument("-s", "--safe", action="store_true", help="save a checkpoint when the program gets interrupted or an error occurs")
    parser.add_argument("-ce", "--checkpoint-every", type=int, default=None, help="export checkpoint every X frame")
    parser.add_argument("-cd", "--checkpoint-end", action="store_true", help="export checkpoint at the last frame")
    parser.add_argument("-nx", "--no-execute", action="store_true", help="do not open the output video file when done")
    parser.add_argument("-re", "--replace", action="store_true", help="overwrite any existing output file (by default, a new filename is generated to avoid conflicts and overwriting)")
    parser.add_argument("-nc", "--no-export-config", action="store_true", help="disable automatic configuration export")
    parser.add_argument("-ef", "--export-flow", action="store_true", help="export computed flow to a file")
    parser.add_argument("-rf", "--round-flow", action="store_true", help="export preprocessed flow as integer values (faster and lighter, but may introduce artefacts)")
    parser.add_argument("-po", "--preview-output", action="store_true", help="preview output while exporting")
    parser.add_argument("-ll", "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level", default="DEBUG")
    parser.add_argument("-lh", "--log-handler", type=str, help="Comma-separated list of log handlers (possible values are 'file', 'stream' or 'null' (default))", default="null")
    parser.add_argument("-lp", "--log-path", type=pathlib.Path, help="Path to log file (if 'file' log handler is used, see --log-handler)", default=pathlib.Path("transflow.log"))

    # GUI Args
    parser.add_argument("-gh", "--gui-host", type=str, default="localhost", help="GUI host address")
    parser.add_argument("-gp", "--gui-port", type=int, default=8000, help="GUI port")
    parser.add_argument("-gm", "--gui-mjpeg-port", type=int, default=8001, help="GUI MJPEG port")

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
                    reset_pixels_leave_healed_spot=d.get("layer_flag_resetleavehealed"),
                    reset_pixels_leave_empty_spot=d.get("layer_flag_resetleaveempty"),
                    reset_random_factor=d.get("layer_reset_random_factor"),
                    reset_constant_step=d.get("layer_reset_constant_step"),
                    reset_linear_factor=d.get("layer_reset_linear_factor"),
                    mask_introduction=d.get("layer_mask_introduction"),
                    introduce_pixels_on_empty_spots=d.get("layer_introduce_empty"),
                    introduce_pixels_on_filled_spots=d.get("layer_introduce_filled"),
                    introduce_moving_pixels=d.get("layer_introduce_moving"),
                    introduce_unmoving_pixels=d.get("layer_introduce_unmoving"),
                    introduce_once=d.get("layer_introduce_once"),
                )
                for d in getattr(args, "layers", [])
            ],
            # Accumulator Args
            # acc_method=args.acc_method,
            # reset_mode=args.reset_mode,
            # reset_alpha=args.reset_alpha,
            # reset_mask_path=args.reset_mask,
            # heatmap_mode=args.heatmap_mode,
            # heatmap_args=args.heatmap_args,
            # heatmap_reset_threshold=args.heatmap_reset_threshold,
            # accumulator_background=args.accumulator_background,
            # stack_composer=args.stack_composer,
            # initial_canvas=args.initial_canvas,
            # bitmap_mask_path=args.bitmap_mask,
            # crumble=args.crumble,
            # bitmap_introduction_flags=args.bitmap_introduction_flags,
            # Output Args
            output_path=args.output,
            vcodec=args.vcodec,
            size=args.size,
            output_intensity=args.output_intensity,
            output_heatmap=args.output_heatmap,
            output_accumulator=args.output_accumulator,
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
