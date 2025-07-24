# Transflow C++

This version of transflow is a limited version of the Python one, only allowing for transferring optical from one webcam stream on the frames from another webcam streams. The optical flow is computed using Gunnar Farneback's algorithm.

## Compilation

1. Download and compile OpenCV using [this guide](https://opencv.org/get-started/) (this might take a while)
2. Compile [transflow.cpp](transflow.cpp) linking OpenCV

> [!NOTE]
> If you compile on something other than Windows, you might want to edit the line 14 to set the correct device API for your system. For instance, Linux users might want to use `cv::CAP_V4L2` instead of `cv::CAP_DSHOW`.

## Usage

> [!NOTE]
> On Windows, you must put OpenCV DLLs in the same directory as the executable. On Linux this should not be an issue.

```console
./transflow.exe <motion device id> <bitmap device id> [-w WIDTH] [-h HEIGHT] [-b BLOCKSIZE] [-r FRAMERATE] [-p PROBABILITY] [-m {off,random,linear}] [-f, --flip]
```

While running, the following key bindings are available:

Key | Function
--- | --------
`ESC` | Exit the program
`+` | Increase pixel reset factor
`-` | Decrease pixel reset factor
`f` | Flip the motion source
`m` | Change reset mode
`s` | Save a screenshot
