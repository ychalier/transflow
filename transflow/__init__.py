"""Set of tools for transferring optical flow from one media to another.
"""

import argparse


def parse_timestamp(timestamp: str | None) -> float | None:
    import re
    if timestamp is None:
        return None
    a = re.match(r"(\d\d):(\d\d):(\d\d)(?:\.(\d\d\d))?", timestamp)
    h = int(a.group(1))
    m = int(a.group(2))
    s = int(a.group(3))
    ms = 0
    if a.group(4) is not None:
        ms = int(a.group(4))
    return 3600 * h + 60 * m + s + ms/1000


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("flow",
        type=str,
        help="input flow: a path to either a video file or a zip archive "\
            "(precomputed flow or checkpoint)")
    parser.add_argument("-f", "--extra-flow",
        type=str,
        nargs="*",
        help="merge multiple flow sources")
    parser.add_argument("-sm", "--flows-merging-function",
        type=str, default="sum",
        choices=["first", "sum", "average", "difference", "product", "maskbin",
                 "masklin", "absmax"],
        help="operations to aggregate extra flow sources")
    parser.add_argument("-b", "--bitmap",
        type=str, default=None,
        help="input bitmap: either a path to a video or an image file or a "\
            "still image generator (color, noise, bwnoise, cnoise, gradient); "\
            "if None, the input flow will be preprocessed")
    parser.add_argument("-o", "--output",
        type=str, default=None,
        help="output path: if provided, path to export the output video (as "\
            "an MP4 file) ; otherwise, opens a temporary display window")
    parser.add_argument("-ba", "--bitmap-alteration",
        type=str, default=None,
        help="path to a PNG file containing alteration to apply to bitmaps")
    parser.add_argument("-ef", "--export-flow",
        action="store_true",
        help="export computed flow to a file")
    parser.add_argument("-oi", "--output-intensity",
        action="store_true",
        help="output flow intensity as a heatmap")
    parser.add_argument("-oh", "--output-heatmap",
        action="store_true",
        help="output heatmap instead of transformed bitmap")
    parser.add_argument("-oa", "--output-accumulator",
        action="store_true",
        help="output accumulator instead of transformed bitmap")
    parser.add_argument("-mv", "--use-mvs",
        action="store_true",
        help="use motion vectors to compute the optical flow instead of "\
            "OpenCV's Farneback algorithm")
    parser.add_argument("-vc", "--vcodec",
        type=str, default="h264",
        help="video codec for the output video file")
    parser.add_argument("-rm", "--reset-mode",
        type=str, choices=["off", "random", "linear"], default="off",
        help="indices re-introduction mode")
    parser.add_argument("-ra", "--reset-alpha",
        type=float, default=.1,
        help="indices re-introduction coefficient (effect my vary depending "\
            "on mode)")
    parser.add_argument("-rk", "--reset-mask",
        type=str, default=None,
        help="path to an image file to use a per-pixel re-introduction "\
            "coefficient")
    parser.add_argument("-hm", "--heatmap-mode",
        type=str, default="discrete", choices=["discrete", "continuous"],
        help="heatmap mode, discrete means binary (motion or not) "\
            "and continuous means flow magnitude")
    parser.add_argument("-ha", "--heatmap-args",
        type=str, default="0:4:2:1",
        help="in discrete mode, args are min:max:add:sub (ints), "\
            "in continuous mode, args are max:decay:treshold (floats)")
    parser.add_argument("-nx", "--no-execute",
        action="store_true",
        help="do not open the output video file when done")
    parser.add_argument("-r", "--replace",
        action="store_true",
        help="overwrite any existing output file (by default, a new filename "\
            "is generated to avoid conflicts and overwriting)")
    parser.add_argument("-fm", "--flow-mask",
        type=str, default=None,
        help="path to an image to applay a per-pixel flow gain")
    parser.add_argument("-fk", "--flow-kernel",
        type=str, default=None,
        help="path to a NPY file storing a convolution kernel to apply on the "\
            "optical flow (impacts performances)")
    parser.add_argument("-cc", "--cv-config",
        type=str, default=None,
        help="path to a JSON file containing settings for OpenCV's Farneback "\
            "algorithm or 'window' to use a live control window; "\
            "if None, a default config is used")
    parser.add_argument("-fg", "--flow-gain",
        type=str, default=None,
        help="multiply the flow at each frame; can be a constant or a Python "\
            "expression (will be passed to the `eval` function) depending on "\
            "the variable `t`, the frame timestamp in seconds")
    parser.add_argument("-rf", "--round-flow",
        action="store_true",
        help="export preprocessed flow as integer values (faster and lighter, "\
            "but may introduce artefacts when combined with flow gain)")
    parser.add_argument("-sz", "--size",
        type=str, default=None,
        help="target video size, for webcams and generated bitmaps, "\
            "of the form WIDTHxHEIGHT")
    parser.add_argument("-m", "--acc-method",
        type=str, default="map", choices=["map", "stack", "sum"],
        help="accumulator method ('map' is default, 'stack' is very slow, "\
            "'sum' only works with backward flows)")
    parser.add_argument("-sb", "--stack-background",
        type=str, default="ffffff",
        help="background color used in stack remapper")
    parser.add_argument("-sc", "--stack-composer",
        type=str, choices=["top", "add", "sub", "avg"], default="top",
        help="stack remapper compose function")
    parser.add_argument("-d", "--direction",
        type=str, choices=["forward", "backward"], default="backward",
        help="direction of flow computation")
    parser.add_argument("-rs", "--render-scale",
        type=float, default=0.1,
        help="render scale for heatmap and accumulator output")
    parser.add_argument("-rc", "--render-colors",
        type=str, default=None,
        help="colors for rendering heatmap (2 colors) and accumulator "\
            "(4 colors) outputs, hex format, separated by commas")
    parser.add_argument("-rb", "--render-binary",
        action="store_true",
        help="1d render will be binary, ie. no gradient will appear")
    parser.add_argument("-ce", "--checkpoint-every",
        type=int, default=None,
        help="export checkpoint every X frame")
    parser.add_argument("-cd", "--checkpoint-end",
        action="store_true",
        help="export checkpoint at the last frame")
    parser.add_argument("-s", "--safe",
        action="store_true",
        help="save a checkpoint when the program gets interrupted "\
            "or an error occurs")
    parser.add_argument("-sd", "--seed",
        type=int, default=None,
        help="random seed for generating bitmaps")
    parser.add_argument("-ss", "--seek",
        type=str, default=None,
        help="start timestamp for flow source")
    parser.add_argument("-bss", "--bitmap-seek",
        type=str, default=None,
        help="start timestamp for bitmap source")
    parser.add_argument("-t", "--duration",
        type=str, default=None,
        help="max output duration")
    parser.add_argument("-to", "--to",
        type=str, default=None,
        help="end timestamp for flow source")
    args = parser.parse_args()
    seek_time = parse_timestamp(args.seek) if args.seek is not None else 0
    duration_time = parse_timestamp(args.duration)
    if duration_time is None and (args.to is not None):
        duration_time = parse_timestamp(args.to) - seek_time
    if duration_time is not None and duration_time < 0:
        raise ValueError(f"Duration must be positive (got {duration_time:f})")
    from .pipeline import transfer
    transfer(
        args.flow,
        args.bitmap,
        args.output,
        extra_flow_paths=args.extra_flow,
        flows_merging_function=args.flows_merging_function,
        use_mvs=args.use_mvs,
        vcodec=args.vcodec,
        reset_mode=args.reset_mode,
        reset_alpha=args.reset_alpha,
        heatmap_mode=args.heatmap_mode,
        heatmap_args=args.heatmap_args,
        execute=not args.no_execute,
        replace=args.replace,
        mask_path=args.flow_mask,
        kernel_path=args.flow_kernel,
        reset_mask_path=args.reset_mask,
        cv_config=args.cv_config,
        flow_gain=args.flow_gain,
        size=args.size,
        acc_method=args.acc_method,
        stack_background=args.stack_background,
        stack_composer=args.stack_composer,
        direction=args.direction,
        round_flow=args.round_flow,
        export_flow=args.export_flow,
        output_intensity=args.output_intensity,
        output_heatmap=args.output_heatmap,
        output_accumulator=args.output_accumulator,
        render_scale=args.render_scale,
        render_colors=args.render_colors,
        render_binary=args.render_binary,
        checkpoint_every=args.checkpoint_every,
        checkpoint_end=args.checkpoint_end,
        safe=args.safe,
        seed=args.seed,
        seek_time=seek_time,
        bitmap_seek_time=parse_timestamp(args.bitmap_seek),
        duration_time=duration_time,
        bitmap_alteration_path=args.bitmap_alteration)
