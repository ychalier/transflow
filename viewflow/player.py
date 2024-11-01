import enum
import math
import os

import av
from av.sidedata.motionvectors import MotionVector
import cv2
import numpy
import pygame

from .reader import BufferedVideoReader


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


@enum.unique
class FlowSource(enum.Enum):
    FARNEBACK = 0
    MOTION_VECTORS = 1


@enum.unique
class Overlay(enum.Enum):
    NONE = 0
    FLOW = 1
    FLOW_MAGNITUDE = 2


@enum.unique
class Output(enum.Enum):
    SOURCE = 0
    DESTINATION = 1
    RECONSTRUCTED = 2


@enum.unique
class DisplayMode(enum.Enum):
    FULL = 0
    LEFT_RIGHT = 1


PICTURE_TYPES = ["NONE", "I", "P", "B", "S", "SI", "SP", "BI"]


def draw_arrow(surface: pygame.Surface, color: tuple[int, int, int],
               start_pos: tuple[int, int], end_pos: tuple[int, int],
               tip: float = 2, shrink: float = 4):
    u = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    norm = math.sqrt(u[0] ** 2 + u[1] ** 2)
    v = (u[0] / norm, u[1] / norm)
    alpha = math.pi / 8
    beta = math.atan2(u[1], u[0])
    A = start_pos[0] + shrink * v[0], start_pos[1] + shrink * v[1]
    B = end_pos[0] - shrink * v[0], end_pos[1] - shrink * v[1]
    C = (
        B[0] + tip * math.cos(beta + math.pi + alpha),
        B[1] + tip * math.sin(beta + math.pi + alpha)
    )
    D = (
        B[0] + tip * math.cos(beta + math.pi - alpha),
        B[1] + tip * math.sin(beta + math.pi - alpha)
    )
    pygame.draw.aaline(surface, color, A, B)
    pygame.draw.aaline(surface, color, B, C)
    pygame.draw.aaline(surface, color, B, D)


def compute_flow(a: av.VideoFrame, b: av.VideoFrame) -> numpy.ndarray:
    # TODO: feature to change parameters
    a_gray = numpy.mean(a.to_ndarray(format="rgb24"), axis=2)
    b_gray = numpy.mean(b.to_ndarray(format="rgb24"), axis=2)
    flow = cv2.calcOpticalFlowFarneback(
        a_gray,
        b_gray,
        None,
        pyr_scale = 0.5,
        levels = 3,
        winsize = 15,
        iterations = 3,
        poly_n = 5,
        poly_sigma = 1.2,
        flags = 0
    )
    return numpy.round(flow)


def compute_magnitude(flow: numpy.ndarray) -> numpy.ndarray:
    m = numpy.clip(numpy.sqrt(numpy.linalg.norm(flow, axis=2)) / 5, 0, 1)
    m = m.reshape((*m.shape, 1))
    c1 = [0, 0, 106]
    c2 = [183, 49, 33]
    return (1 - m) * c1 + m * c2


def convert_frame(frame: av.VideoFrame | numpy.ndarray, size: tuple[int, int],
                  scale: float, origin: tuple[int, int]) -> numpy.ndarray:
    arr = frame
    if isinstance(frame, av.VideoFrame):
        arr = frame.to_ndarray(format="rgb24")
    width, height = size
    origin_x, origin_y = origin
    i0 = math.floor(origin_y)
    i1 = math.ceil(origin_y + height / scale)
    j0 = math.floor(origin_x)
    j1 = math.ceil(origin_x + width / scale)
    rw = math.ceil((j1 - j0) * scale)
    rh = math.ceil((i1 - i0) * scale)
    arr = cv2.resize(
        arr[i0:i1,j0:j1],
        (rw, rh),
        interpolation=cv2.INTER_NEAREST_EXACT)
    return numpy.transpose(arr, axes=(1, 0, 2))


def apply_flow(frame: av.VideoFrame | numpy.ndarray,
               flow: numpy.ndarray) -> numpy.ndarray:
    # TODO: clip flow
    arr = frame
    if isinstance(frame, av.VideoFrame):
        arr = frame.to_ndarray(format="rgb24")
    height, width, depth = arr.shape
    base = numpy.arange(0, height * width * depth, dtype=int)
    flow_flat = flow[:,:,1] * width + flow[:,:,0]
    flow_flat = numpy.repeat(flow_flat, depth).astype(int) * depth
    numpy.put(arr, base + flow_flat, arr.flat, mode="wrap")
    return arr


def extract_flow_mvs(frame: av.VideoFrame) -> numpy.ndarray:
    mvflow = numpy.zeros((frame.height, frame.width, 2), dtype=int)
    vectors = frame.side_data.get("MOTION_VECTORS")
    if vectors is None:
        return mvflow
    for mv in vectors:
        assert mv.source == -1, "Encode with bf=0 and refs=1"
        i0 = mv.src_y - mv.h // 2
        i1 = mv.src_y + mv.h // 2
        j0 = mv.src_x - mv.w // 2
        j1 = mv.src_x + mv.w // 2
        dx = mv.motion_x / mv.motion_scale
        dy = mv.motion_y / mv.motion_scale
        mvflow[i0:i1, j0:j1] = -round(dx), -round(dy)
    return mvflow


def extract_mvs(frame: av.VideoFrame) -> dict[tuple[int, int], MotionVector]:
    vectors = frame.side_data.get("MOTION_VECTORS")
    if vectors is None:
        return {}
    d = {}
    for mv in vectors:
        assert mv.source == -1, "Encode with bf=0 and refs=1"
        i0 = mv.src_y - mv.h // 2
        i1 = mv.src_y + mv.h // 2
        j0 = mv.src_x - mv.w // 2
        j1 = mv.src_x + mv.w // 2
        for i in range(i0, i1):
            for j in range(j0, j1):
                d[(i, j)] = mv
    return d


def format_duration(seconds):
    mins = int(seconds / 60)
    ms = int(1000 * (seconds - int(seconds)))
    secs = int(seconds - 60 * mins)
    return f"{mins:02d}:{secs:02d}.{ms:03d}"


class VideoPlayer:

    def __init__(self, path: str, width: int, height: int, fullscreen: bool = False):
        # Reader attributes
        self.path = path
        self.reader = BufferedVideoReader(self.path)
        self.cursor = 0
        self.src_frame = None
        self.dst_frame = None
        
        # Pygame attributes
        pygame.init()
        self.window = None
        # self.fullwidth = width
        self.width = width # // 2
        self.height = height
        self.origin_x = 0
        self.origin_y = 0
        self.scale = 1
        self.base_scale = 1
        self.font16 = pygame.font.SysFont("Consolas", 16)
        self.font12 = pygame.font.SysFont("Consolas", 12)
        self.dragging = None

        # Display settings
        self.flow_source = FlowSource.FARNEBACK
        self.overlay = Overlay.FLOW
        self.output = Output.DESTINATION
        self.display_mode = DisplayMode.FULL
        self.fullscreen = fullscreen
        
        # Flow attributes
        self.flow = None
        self.flow_magnitude = None
        self.mvs = None
        self.flow_mvs = None
        self.flow_mvs_magnitude = None
        self.reconstructed_frame = None
        self.reconstructed_mvs_frame = None


    def __enter__(self):
        self.window = pygame.display.set_mode(
            (self.width, self.height),
            pygame.FULLSCREEN if self.fullscreen else 0)
        pygame.display.set_caption(os.path.basename(self.path))
        self.reader.start()
        self.reader.wait_until_ready()
        self.scale = self.width / self.reader.width
        self.base_scale = self.scale
        self.set_cursor(0)        
    
    def set_cursor(self, new_cursor: int):
        self.cursor = new_cursor % self.reader.framecount
        self.src_frame = self.reader[self.cursor]
        self.dst_frame = self.reader[(self.cursor + 1) % self.reader.framecount]
        self.flow = compute_flow(self.src_frame, self.dst_frame)
        self.flow_magnitude = compute_magnitude(self.flow)
        self.flow_mvs = extract_flow_mvs(self.src_frame)
        self.flow_mvs_magnitude = compute_magnitude(self.flow_mvs)
        self.reconstructed_frame = apply_flow(self.src_frame, self.flow)
        self.reconstructed_mvs_frame = apply_flow(self.src_frame, self.flow_mvs)
        self.mvs = extract_mvs(self.src_frame)

    def goto_prev(self):
        self.set_cursor(self.cursor - 1)
    
    def goto_next(self):
        self.set_cursor(self.cursor + 1)

    def clamp_origin(self):
        self.origin_x = max(0, min(
            self.reader.width - self.width / self.scale,
            self.origin_x))
        self.origin_y = max(0, min(
            self.reader.height - self.height / self.scale,
            self.origin_y))
    
    def update(self) -> bool:
        ZOOM_INTENSITY = 0.2
        should_draw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_RIGHT:
                    self.goto_next()
                    should_draw = True
                elif event.key == pygame.K_LEFT:
                    self.goto_prev()
                    should_draw = True
                elif event.key == pygame.K_KP_1:
                    self.overlay = Overlay.NONE
                    should_draw = True
                elif event.key == pygame.K_KP_2:
                    self.overlay = Overlay.FLOW
                    should_draw = True
                elif event.key == pygame.K_KP_3:
                    self.overlay = Overlay.FLOW_MAGNITUDE
                    should_draw = True
                elif event.key == pygame.K_KP_4:
                    self.output = Output.SOURCE
                    should_draw = True
                elif event.key == pygame.K_KP_5:
                    self.output = Output.DESTINATION
                    should_draw = True
                elif event.key == pygame.K_KP_6:
                    self.output = Output.RECONSTRUCTED
                    should_draw = True
                elif event.key == pygame.K_KP_7:
                    self.flow_source = FlowSource.FARNEBACK
                    should_draw = True
                elif event.key == pygame.K_KP_8:
                    self.flow_source = FlowSource.MOTION_VECTORS
                    should_draw = True
                elif event.key == pygame.K_y:
                    if self.display_mode == DisplayMode.FULL:
                        self.display_mode = DisplayMode.LEFT_RIGHT
                    elif self.display_mode == DisplayMode.LEFT_RIGHT:
                        self.display_mode = DisplayMode.FULL
                    should_draw = True
            elif event.type == pygame.MOUSEWHEEL:
                should_draw = True
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x >= self.width:
                    mouse_x -= self.width
                wheel = int(event.precise_y)
                zoom = math.exp(wheel * ZOOM_INTENSITY)
                self.origin_x -= mouse_x / (self.scale * zoom) - mouse_x / self.scale
                self.origin_y -= mouse_y / (self.scale * zoom) - mouse_y / self.scale
                self.scale = max(self.base_scale, self.scale * zoom)
                self.clamp_origin()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x >= 8 and mouse_x <= self.width - 8\
                    and mouse_y >= self.height - 24 and mouse_y <= self.height - 8:
                    t = max(0, min(1, (mouse_x - 8) / (self.width - 16)))
                    self.set_cursor(int(t * self.reader.framecount))
                else:
                    self.dragging = (self.origin_x, self.origin_y, mouse_x, mouse_y)
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging is not None:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.origin_x = self.dragging[0] - (mouse_x - self.dragging[2]) / self.scale
                    self.origin_y = self.dragging[1] - (mouse_y - self.dragging[3]) / self.scale
                    self.clamp_origin()
                should_draw = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging = None
        if should_draw:
            self.draw()
        return True
    
    def draw_hud(self):
        # Top left
        text = {
            Overlay.NONE: "ORIGINAL",
            Overlay.FLOW: "OPTICAL FLOW",
            Overlay.FLOW_MAGNITUDE: "OPTICAL FLOW MAGNITUDE"
        }[self.overlay]
        surface = self.font16.render(text, True, WHITE, BLACK)
        self.window.blit(surface, (8, 8))
        h = surface.get_height()
        text = {
            FlowSource.FARNEBACK: "FARNEBACK",
            FlowSource.MOTION_VECTORS: "MOTION VECTORS"
        }[self.flow_source]
        surface = self.font16.render(text, True, WHITE, BLACK)
        self.window.blit(surface, (8, 9 + h))


        if self.display_mode == DisplayMode.LEFT_RIGHT:
            # Top right
            text = {
                Output.SOURCE: "SOURCE FRAME",
                Output.DESTINATION: "DESTINATION FRAME",
                Output.RECONSTRUCTED: "RECONSTRUCTED FRAME",
            }[self.output]
            surface = self.font16.render(text, True, WHITE, BLACK)
            self.window.blit(surface, (self.width - 8 - surface.get_width(), 8))

        # Bottom left
        nbframes = str(self.reader.framecount)
        surface = self.font16.render(
            f"{str(self.cursor + 1).zfill(len(nbframes))}/{nbframes}",
            True, WHITE, BLACK)
        x = 8
        y = self.height - 26 - surface.get_height()
        self.window.blit(surface, (x, y))
        x += surface.get_width() + 4
        surface = self.font16.render(
            format_duration(self.cursor / self.reader.framerate),
            True, WHITE, BLACK)
        self.window.blit(surface, (x, y))
        x += surface.get_width() + 4
        surface = self.font16.render(
            PICTURE_TYPES[int(self.src_frame.pict_type)],
            True, WHITE, BLACK)
        self.window.blit(surface, (x, y))
        
        # Bottom right
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x >= self.width:
            mouse_x -= self.width
        j = int(mouse_x / self.scale + self.origin_x)
        i = int(mouse_y / self.scale + self.origin_y)
        surface = self.font16.render(
            f"x{self.scale:.2f} {j},{i}",
            True, WHITE, BLACK)
        self.window.blit(surface, (self.width - 8 - surface.get_width(), y))

        # Bottom
        pygame.draw.rect(self.window, BLACK, pygame.Rect(8, self.height - 24, self.width - 16, 16))
        pygame.draw.line(self.window, WHITE, (16, self.height - 16), (self.width - 16, self.height - 16), 1)
        curx = 16 + self.cursor / (self.reader.framecount - 1) * (self.width - 32)
        pygame.draw.line(self.window, RED, (curx, self.height - 20), (curx, self.height - 12), 4)

    def draw_frame(self, frame: av.VideoFrame | numpy.ndarray, x: int, w: int, alpha: int = 255):
        surface = pygame.surfarray.make_surface(convert_frame(
            frame,
            (self.width, self.height),
            self.scale,
            (self.origin_x, self.origin_y)
        ))
        if alpha < 255:
            surface = surface.convert_alpha()
            surface.set_alpha(alpha)
        area = pygame.Rect(
            -(int(self.origin_x) - self.origin_x) * self.scale,
            -(int(self.origin_y) - self.origin_y) * self.scale,
            w,
            self.height
        )
        self.window.blit(surface, (x, 0), area)

    def i2c(self, j, i):
        return (j - self.origin_x) * self.scale, (i - self.origin_y) * self.scale
    
    def c2i(self, x, y):
        return x / self.scale + self.origin_x, y / self.scale + self.origin_y

    def draw_flow(self):
        flow_arr = self.flow if self.flow_source == FlowSource.FARNEBACK else self.flow_mvs
        i0 = math.floor(self.origin_y)
        i1 = math.ceil(self.origin_y + self.height / self.scale - .5)
        j0 = math.floor(self.origin_x)
        j1 = math.ceil(self.origin_x + self.width / self.scale - .5)
        n = 20
        stepi = max(1, (i1 - i0) // n)
        stepj = max(1, (j1 - j0) // n)
        offseti = stepi // 2
        offsetj = stepj // 2
        for i in range(i0 + offseti, i1, stepi):
            for j in range(j0 + offsetj, j1, stepj):
                if abs(flow_arr[i][j][0]) < 1 and abs(flow_arr[i][j][1]) < 1:
                    continue
                draw_arrow(
                    self.window,
                    WHITE,
                    self.i2c(j + .5, i + .5),
                    self.i2c(j + flow_arr[i][j][0] + .5, i + flow_arr[i][j][1] + .5),
                    tip=4*math.log(1 + self.scale),
                    shrink=.1 * self.scale,
                )
    
    def draw_cursor(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x >= self.width:
            mouse_x -= self.width
        j0 = int(mouse_x / self.scale + self.origin_x)
        i0 = int(mouse_y / self.scale + self.origin_y)
        if self.flow_source == FlowSource.MOTION_VECTORS:
            if (i0, j0) not in self.mvs:
                return
            mv = self.mvs[(i0, j0)]
            x0, y0 = self.i2c(mv.src_x - .5 * mv.w, mv.src_y - .5 * mv.h)
            x1, y1 = self.i2c(mv.dst_x - .5 * mv.w, mv.dst_y - .5 * mv.h)
            pygame.draw.rect(self.window, BLUE, pygame.Rect(x0, y0, self.scale * mv.w, self.scale * mv.h), 1)
            pygame.draw.rect(self.window, RED, pygame.Rect(x1, y1, self.scale * mv.w, self.scale * mv.h), 1)
            if self.display_mode == DisplayMode.LEFT_RIGHT:
                pygame.draw.rect(self.window, BLUE, pygame.Rect(x0 + self.width // 2, y0, self.scale * mv.w, self.scale * mv.h), 1)
                pygame.draw.rect(self.window, RED, pygame.Rect(x1 + self.width // 2, y1, self.scale * mv.w, self.scale * mv.h), 1)
            return    
        flow_arr = self.flow
        x0, y0 = self.i2c(j0, i0)
        j1 = j0 + int(flow_arr[i0][j0][0])
        i1 = i0 + int(flow_arr[i0][j0][1])
        x1, y1 = self.i2c(j1, i1)
        pygame.draw.rect(self.window, BLUE, pygame.Rect(x0, y0, self.scale + 1, self.scale + 1), 1)
        pygame.draw.rect(self.window, RED, pygame.Rect(x1, y1, self.scale + 1, self.scale + 1), 1)
        if self.display_mode == DisplayMode.LEFT_RIGHT:
            pygame.draw.rect(self.window, BLUE, pygame.Rect(x0 + self.width // 2, y0, self.scale + 1, self.scale + 1), 1)
            pygame.draw.rect(self.window, RED, pygame.Rect(x1 + self.width // 2, y1, self.scale + 1, self.scale + 1), 1)

    def draw(self):
        self.window.fill(BLACK)
        w = self.width if self.display_mode == DisplayMode.FULL else self.width // 2
        self.draw_frame(self.src_frame, 0, w)
        if self.overlay == Overlay.FLOW_MAGNITUDE:
            if self.flow_source == FlowSource.FARNEBACK:
                self.draw_frame(self.flow_magnitude, 0, w, alpha=175)
            elif self.flow_source == FlowSource.MOTION_VECTORS:
                self.draw_frame(self.flow_mvs_magnitude, 0, w, alpha=175)
        if self.overlay in [Overlay.FLOW, Overlay.FLOW_MAGNITUDE]:
            self.draw_flow()
        if self.display_mode == DisplayMode.LEFT_RIGHT:
            right_frame = self.dst_frame
            if self.output == Output.SOURCE:
                right_frame = self.src_frame
            elif self.output == Output.RECONSTRUCTED:
                if self.flow_source == FlowSource.FARNEBACK:
                    right_frame = self.reconstructed_frame
                elif self.flow_source == FlowSource.MOTION_VECTORS:
                    right_frame = self.reconstructed_mvs_frame
            self.draw_frame(right_frame, self.width // 2, w)
        self.draw_cursor()
        self.draw_hud()
        pygame.display.flip()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.terminate()
