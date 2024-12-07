"""Simple GUI for analyzing the mapping from a checkpoint, and slightly alter
the input image to control how the output will look.
"""
import argparse
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


class Window:

    def __init__(self,width: int, ckpt_path: str, bitmap_path: str):

        self.ckpt_path = ckpt_path
        self.bitmap_path = bitmap_path

        pygame.init()
        self.window = None
        self.width = width
        self.height = None

        self.mapping = None
        self.bitmap = None
        self.anchors = None
        self.hover_surfaces = None
        self.anchors_ordered = None
        self.hovering = None
        self.anchor_colors = {}

        self.padding = 8
        self.border_width = 2
        self.square_size = 30
        self.square_padding = 2
        self.anchors_per_row = None
        self.h3 = None

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
            mask = numpy.zeros((*self.bitmap.shape[:2], 4), dtype=numpy.uint8)
            for i, j in targets:
                mask[i,j] = (255, 0, 0, 255)
            array = mask.transpose(1, 0, 2)
            surf = pygame.Surface((854, 480), pygame.SRCALPHA)
            pygame.pixelcopy.array_to_surface(surf, array[:,:,0:3])
            surface_alpha = numpy.array(surf.get_view('A'), copy=False)
            surface_alpha[:,:] = array[:,:,3]
            self.hover_surfaces[anchor] = surf

        self.anchors_ordered = [x[0] for x in sorted(self.anchors.items(), key=lambda x: -len(x[1]))]
        for anchor in self.anchors_ordered:
            self.anchor_colors[anchor] = tuple(self.bitmap[anchor[0], anchor[1]].tolist())

    def __enter__(self):
        self.load()

        h1 = 3 * self.padding
        h2 = self.mapping.shape[0] * ((self.width - 3 * self.padding) // 2) / self.mapping.shape[1]
        self.anchors_per_row = (self.width - self.padding) // (self.square_size + self.square_padding)
        self.h3 = (self.square_size + self.square_padding) * math.ceil(len(self.anchors) / self.anchors_per_row)
        self.height = h1 + h2 + self.h3

        self.window = pygame.display.set_mode(
            (self.width, self.height))
        pygame.display.set_caption(os.path.basename(self.ckpt_path))

    def draw(self):
        self.window.fill(BACKGROUND_COLOR)

        surface_width = (self.width - 3 * self.padding) // 2
        surface_height = surface_width * 9 // 16

        altered_bitmap = numpy.copy(self.bitmap)
        for anchor, color in self.anchor_colors.items():
            altered_bitmap[*anchor] = color

        bitmap_surface = pygame.transform.scale(pygame.surfarray.make_surface(altered_bitmap.transpose(1, 0, 2)), (surface_width, surface_height))

        output = altered_bitmap[self.mapping[:,:,1], self.mapping[:,:,0], :]
        output_surface = pygame.transform.scale(pygame.surfarray.make_surface(output.transpose(1, 0, 2)), (surface_width, surface_height))

        self.window.fill(BORDER_COLOR, (
            self.padding - self.border_width,
            self.height - surface_height - self.padding - self.border_width,
            surface_width + 2 * self.border_width,
            surface_height + 2 * self.border_width))
        self.window.fill(BORDER_COLOR, (
            surface_width + 2 * self.padding - self.border_width,
            self.height - surface_height - self.padding - self.border_width,
            surface_width + 2 * self.border_width,
            surface_height + 2 * self.border_width))

        self.window.blit(bitmap_surface, (
            self.padding,
            self.height - surface_height - self.padding))
        self.window.blit(output_surface, (
            surface_width + 2 * self.padding,
            self.height - surface_height - self.padding))

        anchor_x = anchor_y = self.padding
        for anchor in self.anchors_ordered:
            self.window.fill(self.anchor_colors[anchor], (
                anchor_x,
                anchor_y,
                self.square_size,
                self.square_size))
            anchor_x += self.square_size + self.square_padding
            if anchor_x + self.square_size > self.width - self.padding:
                anchor_x = self.padding
                anchor_y += self.square_size + self.square_padding

        if self.hovering is not None:
            self.window.blit(
                pygame.transform.scale(
                    self.hover_surfaces[self.hovering],
                    (surface_width, surface_height)),
                (surface_width + 2 * self.padding,
                 self.height - surface_height - self.padding))
            pygame.draw.rect(self.window, (255, 0, 0), (
                (self.hovering[1] - 2) / self.mapping.shape[1] * surface_width + self.padding,
                self.height - (self.hovering[0] + 2) / self.mapping.shape[0] * surface_height - self.padding,
                5 * self.mapping.shape[1] / surface_width,
                5 * self.mapping.shape[0] / surface_height), 1)

        pygame.display.flip()

    def get_anchor(self, x: int, y: int) -> tuple[int, int] | None:
        if x > self.padding and x < self.width - self.padding\
            and y > self.padding and y < self.padding + self.h3:
            i = (y - self.padding) // (self.square_size + self.square_padding)
            j = (x - self.padding) // (self.square_size + self.square_padding)
            k = i * self.anchors_per_row + j
            if k < len(self.anchors_ordered):
                return self.anchors_ordered[k]
        return None

    def update(self) -> bool:
        should_draw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                new_hovering = self.get_anchor(mouse_x, mouse_y)
                if new_hovering != self.hovering:
                    should_draw = True
                self.hovering = new_hovering
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                anchor = self.get_anchor(mouse_x, mouse_y)
                if anchor is not None:
                    result = askcolor(self.anchor_colors[anchor])
                    if result is not None:
                        self.anchor_colors[anchor] = result
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