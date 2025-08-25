"""Simple GUI for analyzing the mapping from a checkpoint, and slightly alter
the input image to control how the output will look.

Bindings:
- Left click: change color
- Right click: reset color (can be held down)
- Ctrl+R: reset all colors
- Ctrl+C: store the color currently pointed at in the buffer
- Ctrl+V: apply the buffered color to the region pointed at (can be held down)
- Ctrl+S: export alteration input as PNG
"""
import argparse
import colorsys
import json
import math
import os
import pickle
import threading
import time
import tkinter
import tkinter.colorchooser
import warnings
import zipfile

import numpy
import PIL.Image

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from transflow.utils import parse_color
from transflow.compositor import Compositor
from transflow.compositor.layers.data import DataLayer


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BORDER_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (32, 32, 32)


def ask_color(base_color: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """For some reason, opening/closing the color picker from the main thread
    randomly kills the main window. Doing it in a dedicated thread seems to fix
    this, though I do not understand why.
    """
    class ColorThread(threading.Thread):
        def __init__(self, base_color):
            threading.Thread.__init__(self)
            self.base_color = base_color
            self.result = None, None
        def run(self):
            self.result = tkinter.colorchooser.askcolor(color=self.base_color)
    thread = ColorThread(base_color)
    thread.start()
    thread.join()
    rgb, hex = thread.result
    return rgb


def ask_export_format() -> bool:
    """
    @return True if all sources should be exported
    """
    window = tkinter.Tk()
    window.title("Export")
    label = tkinter.Label(
        window,
        text="Some sources still have their defaults colors."\
            "\nChoose what to export.")
    label.grid(column=0, row=0, columnspan=2)
    vars = {}
    def click_changes():
        vars["choice"] = False
        window.destroy()
    def click_all():
        vars["choice"] = True
        window.destroy()
    tkinter.Button(window, text="Changes only", command=click_changes)\
        .grid(column=0, row=1)
    tkinter.Button(window, text="All", command=click_all)\
        .grid(column=1, row=1)
    window.mainloop()
    return bool(vars.get("choice"))


def get_opposite_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    hue = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)[0]
    R, G, B = colorsys.hls_to_rgb(hue + .5, .5, 1)
    return (int(255 * R), int(255 * G), int(255 * B))


class Window:

    def __init__(self,
            width: int,
            ckpt_path: str,
            pixmap_path: str | None = None,
            max_sources_display: int = 264,
            layer_index: int = 0,
            background_color: str = "#df00ff",
            silent: bool = False):

        self.ckpt_path = ckpt_path
        self.pixmap_path = pixmap_path
        self.layer_index = layer_index
        self.background_color = parse_color(background_color)
        self.silent = silent
        self.flow_path = None
        self.cursor = None

        pygame.init()
        self.window = None
        self.width = width
        self.height = None
        self.font12 = pygame.font.SysFont("Consolas", 12)

        self.w = None
        self.h = None
        self.mapping = None
        self.pixmap = None
        self.alpha = None
        self.masks = None
        self.targets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.sources: list[tuple[int, int]] = []
        self.colors: dict[tuple[int, int], tuple[int, int, int]] = {}
        self.hovered = None
        self.default_colors = {}
        self.buffer = WHITE
        self.is_v_down = False

        self.padding = 8
        self.border_width = 1
        self.square_size = 16
        self.square_padding = 2
        self.sources_per_row = None
        self.height_sources = None
        self.max_sources_display = max_sources_display

    def load(self):

        # Load mapping
        with zipfile.ZipFile(self.ckpt_path) as archive:
            with archive.open("meta.json") as file:
                meta = json.load(file)
                self.flow_path = meta["config"]["flow_path"]
                self.cursor = meta["cursor"]
            with archive.open("compositor.bin") as file:
                compositor: Compositor = pickle.load(file)
        if not compositor.layers or self.layer_index >= len(compositor.layers):
            raise ValueError(f"Compositor does not have enough layers (wants layer index {self.layer_index}, has {len(compositor.layers)} layer(s))")
        layer = compositor.layers[self.layer_index]
        if not isinstance(layer, DataLayer):
            raise ValueError(f"Layer has incorrect class {type(layer)}, must be one of 'moveref' or 'introduction'")
        
        self.mapping = numpy.concatenate(
            [layer.data[:,:,layer.INDEX_J][:,:,numpy.newaxis], layer.data[:,:,layer.INDEX_I][:,:,numpy.newaxis]],
            axis=2).astype(int)
        self.alpha = layer.data[:,:,layer.INDEX_ALPHA]
        self.h = self.mapping.shape[0]
        self.w = self.mapping.shape[1]

        # Identify sources
        self.targets = {}
        for i in range(self.h):
            for j in range(self.w):
                if self.alpha[i][j] != 1:
                    source = (-1, -1)
                else:
                    source = (self.mapping[i][j][1], self.mapping[i][j][0])
                self.targets.setdefault(source, [])
                self.targets[source].append((i, j))
        if not self.silent and len(self.targets) > self.max_sources_display:
            warnings.warn(f"Found too many sources! ({len(self.targets)})")

        # Selecting sources
        sorted_targets = sorted(self.targets.items(), key=lambda x: -len(x[1]))
        self.sources = [x[0] for x in sorted_targets][:self.max_sources_display]

        # Building masks
        self.masks = {}
        for source in self.sources:
            targets = self.targets[source]
            self.alpha = numpy.zeros((self.w, self.h), dtype=numpy.uint8)
            for i, j in targets:
                self.alpha[j,i] = 255
            self.masks[source] = self.alpha

        # Loading or generating pixmap
        if self.pixmap_path is not None:
            self.pixmap = numpy.array(PIL.Image.open(self.pixmap_path))[:,:,:3].astype(numpy.uint8)
        else:
            self.pixmap = numpy.random.randint(0, 255, (*self.mapping.shape[:2], 3), dtype=numpy.uint8)

        # Collecting colors
        for source in self.sources:
            if source == (-1, -1):
                color = self.background_color
            else:
                color = tuple(self.pixmap[source[0], source[1]])
            self.default_colors[source] = color
            self.colors[source] = color

    def __enter__(self):
        self.load()
        assert self.w is not None
        assert self.h is not None

        self.height_panes =\
            self.h * ((self.width - 3 * self.padding) // 2) / self.w
        self.sources_per_row =\
            (self.width - self.padding)\
            // (self.square_size + self.square_padding)
        self.height_sources =\
            (self.square_size + self.square_padding)\
            * math.ceil(len(self.sources) / self.sources_per_row)
        self.height_footer = self.square_size
        self.height =\
            4 * self.padding\
            + self.height_panes + self.height_sources + self.height_footer

        self.surfw = (self.width - 3 * self.padding) // 2
        self.surfh = self.surfw * 9 // 16

        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(os.path.basename(self.ckpt_path))

    def draw(self):
        assert self.window is not None, self.window
        self.window.fill(BACKGROUND_COLOR)

        # Draw Sources
        src_x = src_y = self.padding
        for source in self.sources:
            if source == self.hovered:
                self.window.fill(RED, (
                    src_x - self.border_width,
                    src_y - self.border_width,
                    self.square_size + 2 * self.border_width,
                    self.square_size + 2 * self.border_width
                ))
            self.window.fill(self.colors[source], (
                src_x,
                src_y,
                self.square_size,
                self.square_size))
            src_x += self.square_size + self.square_padding
            if src_x + self.square_size > self.width - self.padding:
                src_x = self.padding
                src_y += self.square_size + self.square_padding

        # Draw Panes
        assert self.height_sources is not None, self.height_sources
        paney = self.height_sources + 2 * self.padding

        assert self.pixmap is not None
        # Draw Left Pane
        altered_pixmap = numpy.copy(self.pixmap)
        for source, color in self.colors.items():
            altered_pixmap[*source] = color
        pixmap_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(altered_pixmap.transpose(1, 0, 2)),
            (self.surfw, self.surfh))
        self.window.fill(BORDER_COLOR, (
            self.padding - self.border_width,
            paney - self.border_width,
            self.surfw + 2 * self.border_width,
            self.surfh + 2 * self.border_width))
        self.window.blit(pixmap_surface, (self.padding, paney))

        # Draw Right Pane
        assert self.mapping is not None
        output = altered_pixmap[self.mapping[:,:,1], self.mapping[:,:,0], :]
        output_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(output.transpose(1, 0, 2)),
            (self.surfw, self.surfh))
        
        assert self.w is not None, self.w
        assert self.h is not None, self.h
        assert self.masks is not None, self.masks        

        self.window.fill(BORDER_COLOR, (
            self.surfw + 2 * self.padding - self.border_width,
            paney - self.border_width,
            self.surfw + 2 * self.border_width,
            self.surfh + 2 * self.border_width))
        self.window.blit(output_surface, (self.surfw + 2 * self.padding, paney))
        
        if (-1, -1) in self.colors:
            surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            surf.fill(self.colors[(-1, -1)], (0, 0, self.w, self.h))
            numpy.array(surf.get_view('A'), copy=False)[:,:] = self.masks[(-1, -1)]
            self.window.blit(pygame.transform.scale(surf, (self.surfw, self.surfh)), (self.surfw + 2 * self.padding, paney))

        # Draw Over Panes
        if self.hovered is not None:
            surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            surf.fill(
                get_opposite_color(self.colors[self.hovered]),
                (0, 0, self.w, self.h))
            numpy.array(surf.get_view('A'), copy=False)[:,:] =\
                self.masks[self.hovered]
            self.window.blit(
                pygame.transform.scale(
                    surf,
                    (self.surfw, self.surfh)),
                (self.surfw + 2 * self.padding, paney))
            pygame.draw.rect(self.window, RED, (
                (self.hovered[1] - 1) / self.w * self.surfw + self.padding,
                paney + (self.hovered[0] - 1) / self.h * self.surfh,
                5 * self.w / self.surfw,
                5 * self.h / self.surfh), 1)

        # Draw Footer
        footery = self.height_sources + self.height_panes + 3 * self.padding
        self.window.fill(self.buffer, (
            self.padding,
            footery,
            self.square_size,
            self.square_size))
        self.draw_hovered_color()

        if self.hovered is not None:
            area = len(self.targets[self.hovered]) / self.h / self.w
            surface = self.font12.render(
                f"Source at ({self.hovered[1]}, {self.hovered[0]}),"\
                    f" {len(self.targets[self.hovered])}px"\
                    f" ({int(100 * area)}% area),"\
                    f" rgb{tuple(map(int, self.colors[self.hovered]))}",
                True, WHITE, BACKGROUND_COLOR)
            self.window.blit(surface, (
                self.width - surface.get_width() - self.padding,
                footery + self.square_size - surface.get_height()))

        pygame.display.flip()

    def get_hovered_color(self) -> tuple[int, int, int]:
        assert self.window is not None, self.window
        mouse_x, mouse_y = pygame.mouse.get_pos()
        color = self.window.get_at((mouse_x, mouse_y))
        return (color.r, color.g, color.b)

    def draw_hovered_color(self, flip=False):
        assert self.window is not None, self.window
        assert self.height_sources is not None
        self.window.fill(self.get_hovered_color(), (
            self.padding + self.square_padding + self.square_size,
            self.height_sources + self.height_panes + 3 * self.padding,
            self.square_size,
            self.square_size))
        if flip:
            pygame.display.flip()

    def get_source(self, x: int, y: int) -> tuple[int, int] | None:
        assert self.height_sources is not None, self.height_sources
        assert self.sources_per_row is not None, self.sources_per_row
        assert self.height is not None, self.height
        assert self.w is not None, self.w
        assert self.h is not None, self.h
        assert self.surfw is not None, self.surfw
        assert self.surfh is not None, self.surfh
        assert self.mapping is not None
        if x > self.padding and x < self.width - self.padding\
            and y > self.padding and y < self.padding + self.height_sources:
            i = (y - self.padding) // (self.square_size + self.square_padding)
            j = (x - self.padding) // (self.square_size + self.square_padding)
            k = i * self.sources_per_row + j
            if k < len(self.sources):
                return self.sources[k]
        if x < 2 * self.padding + self.surfw:
            return None
        if x > 2 * self.padding + 2 * self.surfw:
            return None
        if y < self.height_sources + 2 * self.padding:
            return None
        if y > self.height - self.padding:
            return None
        x -= 2 * self.padding + self.surfw
        y -= self.height_sources + 2 * self.padding
        x = int(x * self.w / self.surfw)
        y = int(y * self.h / self.surfh)
        if y >= self.h or x >= self.w:
            return None
        source = (
            self.mapping[y, x, 1],
            self.mapping[y, x, 0])
        if source in self.sources:
            return source
        return None

    def export(self, force_export_all: bool = False):
        assert self.flow_path is not None, self.flow_path
        assert self.mapping is not None, self.mapping
        path = os.path.join(
            os.path.dirname(self.ckpt_path),
            os.path.splitext(os.path.basename(self.flow_path))[0]
                + f"_{self.cursor:05d}_{int(1000*time.time())}.png")
        has_default_color = False
        for source in self.sources:
            if self.colors[source] == self.default_colors[source]:
                has_default_color = True
                break
        output_all = True
        if has_default_color:
            output_all = force_export_all or ask_export_format()
        array = numpy.zeros((*self.mapping.shape[:2], 4), dtype=numpy.uint8)
        for source in self.sources:
            if not output_all\
                and self.colors[source] == self.default_colors[source]:
                continue
            if source == (-1, -1):
                continue
            array[source[0], source[1]] = (*self.colors[source], 255)
        PIL.Image.fromarray(array).save(path)
        if not self.silent:
            print(f"Exported to {path}")

    def over_buffer(self, x: int, y: int) -> bool:
        assert self.height is not None, self.height
        return x >= self.padding\
            and x < self.padding + self.square_size\
            and y >= self.height - self.padding - self.square_size\
            and y <= self.height - self.padding

    def update(self) -> bool:
        should_draw = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_s\
                    and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.export()
                elif event.key == pygame.K_c\
                    and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.buffer = self.get_hovered_color()
                    should_draw = True
                elif event.key == pygame.K_v:
                    self.is_v_down = True
                    if pygame.key.get_mods() & pygame.KMOD_CTRL\
                        and self.hovered is not None:
                        self.colors[self.hovered] = self.buffer
                        should_draw = True
                elif event.key == pygame.K_r:
                    for source, color in self.default_colors.items():
                        self.colors[source] = color
                    should_draw = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_v:
                    self.is_v_down = False
                elif event.key == pygame.K_LCTRL:
                    self.is_ctrl_down = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                hn = self.get_source(mouse_x, mouse_y)
                if hn != self.hovered:
                    if hn is not None:
                        if pygame.mouse.get_pressed()[2]:
                            self.colors[hn] = self.default_colors[hn]
                        elif self.is_v_down\
                            and pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.colors[hn] = self.buffer
                    should_draw = True
                self.hovered = hn
                if not should_draw:
                    self.draw_hovered_color(True)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                source = self.get_source(mouse_x, mouse_y)
                if source is not None:
                    if event.button == pygame.BUTTON_LEFT:
                        result = ask_color(self.colors[source])
                        if result is not None:
                            self.colors[source] = result
                            should_draw = True
                    elif event.button == pygame.BUTTON_RIGHT:
                        self.colors[source] = self.default_colors[source]
                        should_draw = True
                elif self.over_buffer(mouse_x, mouse_y):
                    result = ask_color(self.buffer)
                    if result is not None:
                        self.buffer = result
                        should_draw = True
        if should_draw:
            self.draw()
        return True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_path",
        type=str,
        help="path to checkpoint file")
    parser.add_argument("pixmap_path",
        type=str, default=None, nargs="?",
        help="path to image file")
    parser.add_argument("-m", "--max-sources-display",
        type=int, default=264,
        help="maximum number of sources to display")
    parser.add_argument("-w", "--width",
        type=int, default=1600,
        help="window width")
    parser.add_argument("-l", "--layer-index", type=int, default=0)
    parser.add_argument("-b", "--background-color", type=str, default="#df00ff")
    args = parser.parse_args()
    window = Window(
        args.width,
        args.ckpt_path,
        args.pixmap_path,
        args.max_sources_display,
        args.layer_index,
        args.background_color)
    with window:
        try:
            window.draw()
            while window.update():
                time.sleep(.001)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()