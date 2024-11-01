# Transflow

Set of tools for transferring optical flow from one media to another.

This tool extracts [optical flow](https://en.wikipedia.org/wiki/Optical_flow) as a dense velocity field from a video input (file or stream) and applies it to an image, a video file or stream, for creative purposes. Multiple techniques can be used, various parameters can be tuned to precisely craft visual effects directly from raw raster data. Here is how it looks (click to view in full resolution):

Flow Source | Bitmap Source | Result
----------- | ------------- | ------
[![River.mp4](assets/River.gif)](assets/River.mp4) | [![Deer.jpg](assets/Deer.jpg)](assets/Deer.jpg) | [![Output.mp4](assets/Output.gif)](assets/Output.mp4)

Have a look at:
- [The web demo](https://chalier.fr/transflow/)
- [The reference guide](USAGE.md)

## Getting Started

### Prerequisites

You'll need a working installation of [Python 3](https://www.python.org/). You'll also need [FFmpeg](https://ffmpeg.org/) binaries, available in `PATH`, the `ffmpeg -version` command should work.

### Basic Installation

The process should be straightforward.

1. Download the [latest release](https://github.com/ychalier/transflow/releases)
2. Install it with `pip`:
   ```console
   pip install ~/Downloads/transflow-1.0.0.tar.gz
   ```
3. Test installation:
   ```console
   python -m transflow --help
   ```

### Alternative Installation

If you want to access the code.

1. Clone or [download](https://github.com/ychalier/transflow/archive/refs/heads/main.zip) this repository:
   ```console
   git clone https://github.com/ychalier/transflow.git
   cd transflow
   ```
2. Install requirements:
   ```console
   pip install -r requirements.txt
   ```
3. Test installation:
   ```console
   python transflow.py --help
   ```

### Usage

The simplest process consists in taking the motion from a video file and applying it to an image:

```console
transflow flow.mp4 -b image.jpg -o output.mp4
```

The alias `transflow` represents either `python -m transflow` or `python transflow.py`, depending on the chosen installation method. For more details, see [USAGE.md](USAGE.md).

## Additional Resources

This repository also contains two other versions of this program:

- in the [cpp](cpp) folder, a C++ version using OpenCV to transfer the flow from one webcam to another,
- in the [www](www) folder, a WebGL version that emulates the effect in a web browser; a version is hosted on [chalier.fr/transflow](https://chalier.fr/transflow/).

There is also another module, [viewflow](viewflow), for visualizing and inspecting optical flow in a video player setting.

## Contributing

Contributions are welcomed. Do not hesitate to submit a pull request with your changes! Submit bug reports and feature suggestions in the [issue tracker](https://github.com/ychalier/transflow/issues/new/choose).

## License

This project is licensed under the GPL-3.0 license.
