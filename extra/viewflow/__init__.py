import argparse
import json
import os
import subprocess
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from .player import VideoPlayer


def get_video_size(path: str) -> tuple[int, int]:
    stdout = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
        "-show_streams", path])
    data = json.loads(stdout)
    width = height = None
    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            width = stream["width"]
            height = stream["height"]
            break
    return width, height


def viewflow(video_path: str, fullscreen: bool = False):
    player = VideoPlayer(video_path, *get_video_size(video_path), fullscreen)
    with player:
        try:
            player.draw()
            while player.update():
                time.sleep(.001)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("-f", "--fullscreen", action="store_true")
    args = parser.parse_args()
    viewflow(args.video_path, args.fullscreen)
