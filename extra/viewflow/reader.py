import threading
import time

import av


def circular_interval(size: int, begin: int, end: int) -> set[int]:
    begin %= size
    end %= size
    if begin > end:
        return set(range(begin, size)).union(range(end + 1))
    return set(range(begin, end + 1))


class VideoReader:

    def __init__(self, path: str):
        self.path = path
        self.container = None
        self.cursor = None
        self.width = None
        self.height = None
        self.framerate = None
        self.timebase = None
        self.framecount = None
        self.frame = None
    
    def open(self):
        self.container = av.open(file=self.path)
        video_ctx = self.container.streams.video[0]
        video_ctx.codec_context.export_mvs = True
        self.frame = next(self.container.decode(video=0))
        self.width = self.frame.width
        self.height = self.frame.height
        self.framerate = float(video_ctx.average_rate)
        self.timebase = float(video_ctx.time_base)
        self.framecount = video_ctx.frames
        self.cursor = 0
    
    def read_frame(self, i=None) -> av.VideoFrame:
        if i is None:
            i = self.cursor
        i %= self.framecount
        if i == self.cursor:
            return self.frame
        if i == self.cursor + 1:
            return self.next()
        return self.seek(i)

    def seek(self, i: int) -> av.VideoFrame:
        """
        @see https://github.com/PyAV-Org/PyAV/discussions/1113
        """
        t = int(i / self.framerate)
        self.container.seek(t * 1000000, backward=True)
        self.frame = next(self.container.decode(video=0))
        j = int(self.frame.pts * self.timebase * self.framerate)
        self.cursor = j
        for _ in range(j, i):
            self.next()
        return self.frame
    
    def next(self) -> av.VideoFrame:
        if self.cursor >= self.framecount:
            raise ValueError("Incorrect cursor value %d" % self.cursor)
        self.cursor += 1
        if self.cursor >= self.framecount - 1:
            self.cursor = 0
            self.container.seek(0)
        self.frame = next(self.container.decode(video=0))
        return self.frame
    
    def close(self):
        self.container.close()


class BufferedVideoReader(VideoReader, threading.Thread):

    def __init__(self, path, before=30, after=30, margin=30):
        threading.Thread.__init__(self, daemon=True)
        VideoReader.__init__(self, path)
        self.before = before
        self.after = after
        self.size = self.before + self.after + 1
        self.buffer: dict[int, av.VideoFrame] = {}
        self.center = None
        self.running = True
        self.changed = False
        self.buffered_frames = 0
        self.margin = margin
        self.lock_ready = threading.Lock()
        self.lock_ready.acquire()

    def setup(self):
        self.open()
        if self.size > self.framecount:
            if self.framecount % 2 == 0:
                self.before = self.framecount // 2 - 1
                self.after = self.framecount // 2
            else:
                self.before = self.framecount // 2
                self.after = self.framecount // 2
        self.center = 0
        self.changed = True
        self.update_buffer()
        self.lock_ready.release()
    
    def wait_until_ready(self):
        self.lock_ready.acquire()

    def interval(self):
        return circular_interval(self.framecount, self.center - self.before, self.center + self.after)

    def in_interval(self, i):
        begin = (self.center - self.before) % self.framecount
        end = (self.center + self.after) % self.framecount
        if begin > end:
            return i >= begin or i <= end + 1
        return i >= begin and i <= end + 1

    def update_buffer(self):
        self.changed = False
        interval = self.interval()
        if self.buffered_frames > self.size + self.margin:
            indices_to_delete = set(self.buffer.keys()).difference(interval)
            for i in indices_to_delete:
                del self.buffer[i]
                self.buffered_frames -= 1
        indices_to_add = interval.difference(self.buffer.keys())
        for i in sorted(indices_to_add):
            # If a frame outside buffer range is accessed while the buffer is
            # filling, we can abort the current operation: next frames will be
            # useless.
            if self.changed and not self.in_interval(self.center):
                return
            self.buffer[i] = self.read_frame(i)
            self.buffered_frames += 1
    
    def __getitem__(self, i: int) -> av.VideoFrame:
        self.changed = i != self.center
        self.center = i
        while not i in self.buffer and self.running:
            time.sleep(.001) # TODO: consider using a lock?
        return self.buffer[i]
    
    def terminate(self):
        self.running = False

    def run(self):
        self.setup()
        while self.running:
            if self.changed:
                self.update_buffer()
            else:
                time.sleep(.001)
        self.close()