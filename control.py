"""Simple GUI for analyzing the mapping from a checkpoint, and slightly alter
the input image to control how the output will look.

Bindings:
- Left click: change color
- Right click: reset color (can be held down)
- Ctrl+R: reset all colors
- Ctrl+C: store the color currently pointed at in the buffer
- Ctrl+V: apply the buffered color to the region pointed at (can be held down)
- Ctrl+S: export altered input as PNG
"""
import argparse
import colorsys
import math
import os
import pickle
import threading
import time
import tkinter.colorchooser
import zipfile

import numpy
import PIL.Image

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from transflow.accumulator import MappingAccumulator


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BORDER_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (32, 32, 32)


def askcolor(base_color: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """For some reason, opening/closing the color picker from the main thread
    randomly kills the main window. Doing it in a dedicated thread seems to fix
    this, though I do not understand why.
    """
    class ColorThread(threading.Thread):
        def __init__(self, base_color):
            threading.Thread.__init__(self)
            self.base_color = base_color
            self.result = None
        def run(self):
            self.result = tkinter.colorchooser.askcolor(color=self.base_color)
    thread = ColorThread(base_color)
    thread.start()
    thread.join()
    rgb, hex = thread.result
    return rgb


def get_opposite_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return tuple(map(
        lambda x: int(255 * x),
        colorsys.hls_to_rgb(h+.5, .5, 1)))


class Window:

    def __init__(self,width: int, ckpt_path: str, bitmap_path: str):

        self.ckpt_path = ckpt_path
        self.bitmap_path = bitmap_path

        pygame.init()
        self.window = None
        self.width = width
        self.height = None
        self.font12 = pygame.font.SysFont("Consolas", 12)

        self.mapping = None
        self.bitmap = None
        self.anchors = None
        self.hover_surfaces = None
        self.anchors_ordered = None
        self.hovering = None
        self.anchor_colors = {}
        self.default_anchor_colors = {}
        self.buffer = WHITE
        self.is_v_down = False

        self.padding = 8
        self.border_width = 1
        self.square_size = 16
        self.square_padding = 2
        self.anchors_per_row = None
        self.height_anchors = None

    def load(self):
        with zipfile.ZipFile(self.ckpt_path) as archive:
            with archive.open("accumulator.bin") as file:
                accumulator = pickle.load(file)
        if not isinstance(accumulator, MappingAccumulator):
            raise ValueError("Checkpoint must contain an accumulator of type"\
                            f"MappingAccumulator, not {type(accumulator)}")
        self.mapping = numpy.concatenate(
            [accumulator.mapx[:,:,numpy.newaxis],
             accumulator.mapy[:,:,numpy.newaxis]],
            axis=2).astype(int)
        self.anchors = {}
        for i in range(self.mapping.shape[0]):
            for j in range(self.mapping.shape[1]):
                anchor = (self.mapping[i][j][1], self.mapping[i][j][0])
                self.anchors.setdefault(anchor, [])
                self.anchors[anchor].append((i, j))
        print("Found", len(self.anchors), "anchors")
        self.bitmap = numpy.array(PIL.Image.open(self.bitmap_path))

        self.hover_surfaces = {}
        for anchor, targets in self.anchors.items():
            alpha = numpy.zeros(self.bitmap.shape[:2], dtype=numpy.uint8)
            for i, j in targets:
                alpha[i,j] = 255
            self.hover_surfaces[anchor] = alpha.T

        self.anchors_ordered = [x[0] for x in sorted(self.anchors.items(), key=lambda x: -len(x[1]))]
        for anchor in self.anchors_ordered:
            self.default_anchor_colors[anchor] = tuple(self.bitmap[anchor[0], anchor[1]].tolist())
            self.anchor_colors[anchor] = self.default_anchor_colors[anchor]

    def __enter__(self):
        self.load()

        self.height_panes = self.mapping.shape[0] * ((self.width - 3 * self.padding) // 2) / self.mapping.shape[1]
        self.anchors_per_row = (self.width - self.padding) // (self.square_size + self.square_padding)
        self.height_anchors = (self.square_size + self.square_padding) * math.ceil(len(self.anchors) / self.anchors_per_row)
        self.height_footer = self.square_size
        self.height = 4 * self.padding + self.height_panes + self.height_anchors + self.height_footer

        self.surface_width = (self.width - 3 * self.padding) // 2
        self.surface_height = self.surface_width * 9 // 16

        self.window = pygame.display.set_mode(
            (self.width, self.height))
        pygame.display.set_caption(os.path.basename(self.ckpt_path))

    def draw(self):
        self.window.fill(BACKGROUND_COLOR)

        # Draw Anchors
        anchor_x = anchor_y = self.padding
        for anchor in self.anchors_ordered:
            if anchor == self.hovering:
                self.window.fill(RED, (
                    anchor_x - self.border_width,
                    anchor_y - self.border_width,
                    self.square_size + 2 * self.border_width,
                    self.square_size + 2 * self.border_width
                ))
            self.window.fill(self.anchor_colors[anchor], (
                anchor_x,
                anchor_y,
                self.square_size,
                self.square_size))
            anchor_x += self.square_size + self.square_padding
            if anchor_x + self.square_size > self.width - self.padding:
                anchor_x = self.padding
                anchor_y += self.square_size + self.square_padding
        
        # Draw Panes
        altered_bitmap = numpy.copy(self.bitmap)
        for anchor, color in self.anchor_colors.items():
            altered_bitmap[*anchor] = color
        bitmap_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(altered_bitmap.transpose(1, 0, 2)),
            (self.surface_width, self.surface_height))
        output = altered_bitmap[self.mapping[:,:,1], self.mapping[:,:,0], :]
        output_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(output.transpose(1, 0, 2)),
            (self.surface_width, self.surface_height))
        paney = self.height_anchors + 2 * self.padding
        self.window.fill(BORDER_COLOR, (
            self.padding - self.border_width,
            paney - self.border_width,
            self.surface_width + 2 * self.border_width,
            self.surface_height + 2 * self.border_width))
        self.window.fill(BORDER_COLOR, (
            self.surface_width + 2 * self.padding - self.border_width,
            paney - self.border_width,
            self.surface_width + 2 * self.border_width,
            self.surface_height + 2 * self.border_width))
        self.window.blit(bitmap_surface, (
            self.padding,
            paney))
        self.window.blit(output_surface, (
            self.surface_width + 2 * self.padding,
            paney))

        # Draw Over Panes
        if self.hovering is not None:
            size = (self.mapping.shape[1], self.mapping.shape[0])
            surf = pygame.Surface(size, pygame.SRCALPHA)
            surf.fill(get_opposite_color(self.anchor_colors[self.hovering]), (0, 0, *size))
            numpy.array(surf.get_view('A'), copy=False)[:,:] = self.hover_surfaces[self.hovering]
            self.window.blit(
                pygame.transform.scale(
                    surf,
                    (self.surface_width, self.surface_height)),
                (self.surface_width + 2 * self.padding, paney))
            pygame.draw.rect(self.window, RED, (
                (self.hovering[1] - 1) / self.mapping.shape[1] * self.surface_width + self.padding,
                paney + (self.hovering[0] - 1) / self.mapping.shape[0] * self.surface_height,
                5 * self.mapping.shape[1] / self.surface_width,
                5 * self.mapping.shape[0] / self.surface_height), 1)
            
        # Draw Footer
        footery = self.height_anchors + self.height_panes + 3 * self.padding
        self.window.fill(self.buffer, (
            self.padding,
            footery,
            self.square_size,
            self.square_size))
        self.draw_hovered_color()

        if self.hovering is not None:
            area = len(self.anchors[self.hovering]) / self.mapping.shape[0] / self.mapping.shape[1]
            surface = self.font12.render(
                f"Anchor at ({self.hovering[1]}, {self.hovering[0]}),"\
                    f" {len(self.anchors[self.hovering])}px"\
                    f" ({int(100 * area)}% area),"\
                    f" rgb{self.anchor_colors[self.hovering]}",
                True, WHITE, BACKGROUND_COLOR)
            w = surface.get_width()
            h = surface.get_height()
            self.window.blit(surface, (self.width - w - self.padding, footery + self.square_size - h))

        pygame.display.flip()

    def get_hovered_color(self) -> tuple[int, int, int]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return self.window.get_at((mouse_x, mouse_y))[:3]

    def draw_hovered_color(self, flip=False):
        self.window.fill(self.get_hovered_color(), (
            self.padding + self.square_padding + self.square_size,
            self.height_anchors + self.height_panes + 3 * self.padding,
            self.square_size,
            self.square_size))
        if flip:
            pygame.display.flip()

    def get_anchor(self, x: int, y: int) -> tuple[int, int] | None:
        if x > self.padding and x < self.width - self.padding\
            and y > self.padding and y < self.padding + self.height_anchors:
            i = (y - self.padding) // (self.square_size + self.square_padding)
            j = (x - self.padding) // (self.square_size + self.square_padding)
            k = i * self.anchors_per_row + j
            if k < len(self.anchors_ordered):
                return self.anchors_ordered[k]
        return self.get_anchor_from_output(x, y)

    def get_anchor_from_output(self, x: int, y: int) -> tuple[int, int] | None:
        if x < 2 * self.padding + self.surface_width:
            return None
        if x > 2 * self.padding + 2 * self.surface_width:
            return None
        if y < self.height_anchors + 2 * self.padding:
            return None
        if y > self.height - self.padding:
            return None
        x -= 2 * self.padding + self.surface_width
        y -= self.height_anchors + 2 * self.padding
        x *= self.mapping.shape[1] / self.surface_width
        y *= self.mapping.shape[0] / self.surface_height
        if int(y) >= self.mapping.shape[0] or int(x) >= self.mapping.shape[1]:
            return None
        return self.mapping[int(y), int(x), 1], self.mapping[int(y), int(x), 0]

    def export(self):
        f = lambda s: os.path.splitext(os.path.basename(s))[0]
        filename = f"{f(self.bitmap_path)}_{f(self.ckpt_path)}_{int(1000*time.time())}.png"
        altered_bitmap = numpy.copy(self.bitmap)
        for anchor, color in self.anchor_colors.items():
            altered_bitmap[*anchor] = color
        PIL.Image.fromarray(altered_bitmap).save(filename)
        print(f"Exported to {os.path.realpath(filename)}")
    
    def is_in_buffer(self, x: int, y: int) -> bool:
        return x >= self.padding and x < self.padding + self.square_size and y >= self.height - self.padding - self.square_size and y <= self.height - self.padding

    def update(self) -> bool:
        should_draw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.export()
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.buffer = self.get_hovered_color()
                    should_draw = True
                elif event.key == pygame.K_v:
                    self.is_v_down = True
                    if pygame.key.get_mods() & pygame.KMOD_CTRL and self.hovering is not None:
                        self.anchor_colors[self.hovering] = self.buffer
                        should_draw = True
                elif event.key == pygame.K_r:
                    for anchor, color in self.default_anchor_colors.items():
                        self.anchor_colors[anchor] = color
                    should_draw = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_v:
                    self.is_v_down = False
                elif event.key == pygame.K_LCTRL:
                    self.is_ctrl_down = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                new_hovering = self.get_anchor(mouse_x, mouse_y)
                if new_hovering != self.hovering:
                    if new_hovering is not None:
                        if pygame.mouse.get_pressed()[2]:
                            self.anchor_colors[new_hovering] = self.default_anchor_colors[new_hovering]
                        elif self.is_v_down and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.anchor_colors[new_hovering] = self.buffer
                    should_draw = True
                self.hovering = new_hovering
                if not should_draw:
                    self.draw_hovered_color(True)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                anchor = self.get_anchor(mouse_x, mouse_y)
                if anchor is not None:
                    if event.button == pygame.BUTTON_LEFT:
                        result = askcolor(self.anchor_colors[anchor])
                        if result is not None:
                            self.anchor_colors[anchor] = result
                            should_draw = True
                    elif event.button == pygame.BUTTON_RIGHT:
                        self.anchor_colors[anchor] = self.default_anchor_colors[anchor]
                        should_draw = True
                elif self.is_in_buffer(mouse_x, mouse_y):
                    result = askcolor(self.buffer)
                    if result is not None:
                        self.buffer = result
                        should_draw = True
        if should_draw:
            self.draw()
        return True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


def start(ckpt_path, bitmap_path):
    window = Window(1600, ckpt_path, bitmap_path)
    with window:
        try:
            window.draw()
            while window.update():
                time.sleep(.001)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_path", type=str, help="path to checkpoint file")
    parser.add_argument("bitmap_path", type=str, help="path to image file")
    args = parser.parse_args()
    start(args.ckpt_path, args.bitmap_path)


if __name__ == "__main__":
    main()