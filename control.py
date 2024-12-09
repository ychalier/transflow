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

from transflow.accumulator import MappingAccumulator


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
            self.result = None
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
    return vars.get("choice")


def get_opposite_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return tuple(map(
        lambda x: int(255 * x),
        colorsys.hls_to_rgb(h+.5, .5, 1)))


class Window:

    def __init__(self, width: int, ckpt_path: str, bitmap_path: str | None,
                 max_sources_display: int = 264):

        self.ckpt_path = ckpt_path
        self.bitmap_path = bitmap_path
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
        self.bitmap = None
        self.targets = None
        self.masks = None
        self.sources = None
        self.colors = {}
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
                self.flow_path = meta["flow_path"]
                self.cursor = meta["cursor"]
            with archive.open("accumulator.bin") as file:
                acc = pickle.load(file)
        if not isinstance(acc, MappingAccumulator):
            raise ValueError("Checkpoint must contain an accumulator of type"\
                            f"MappingAccumulator, not {type(acc)}")
        self.mapping = numpy.concatenate(
            [acc.mapx[:,:,numpy.newaxis], acc.mapy[:,:,numpy.newaxis]],
            axis=2).astype(int)
        self.h = self.mapping.shape[0]
        self.w = self.mapping.shape[1]

        # Identify sources
        self.targets = {}
        for i in range(self.h):
            for j in range(self.w):
                source = (self.mapping[i][j][1], self.mapping[i][j][0])
                self.targets.setdefault(source, [])
                self.targets[source].append((i, j))
        if len(self.targets) > self.max_sources_display:
            warnings.warn(f"Found too many sources! ({len(self.targets)})")

        # Selecting sources
        sorted_targets = sorted(self.targets.items(), key=lambda x: -len(x[1]))
        self.sources = [x[0] for x in sorted_targets][:self.max_sources_display]

        # Building masks
        self.masks = {}
        for source in self.sources:
            targets = self.targets[source]
            alpha = numpy.zeros((self.w, self.h), dtype=numpy.uint8)
            for i, j in targets:
                alpha[j,i] = 255
            self.masks[source] = alpha

        # Loading or generating bitmap
        if self.bitmap_path is not None:
            self.bitmap = numpy.array(PIL.Image.open(self.bitmap_path))[:,:,:3]
        else:
            self.bitmap = numpy.random.randint(0, 255, (*self.mapping.shape[:2], 3))

        # Collecting colors
        for source in self.sources:
            color = tuple(self.bitmap[source[0], source[1]].tolist())
            self.default_colors[source] = color
            self.colors[source] = color

    def __enter__(self):
        self.load()

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
        paney = self.height_sources + 2 * self.padding

        # Draw Left Pane
        altered_bitmap = numpy.copy(self.bitmap)
        for source, color in self.colors.items():
            altered_bitmap[*source] = color
        bitmap_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(altered_bitmap.transpose(1, 0, 2)),
            (self.surfw, self.surfh))
        self.window.fill(BORDER_COLOR, (
            self.padding - self.border_width,
            paney - self.border_width,
            self.surfw + 2 * self.border_width,
            self.surfh + 2 * self.border_width))
        self.window.blit(bitmap_surface, (self.padding, paney))

        # Draw Right Pane
        output = altered_bitmap[self.mapping[:,:,1], self.mapping[:,:,0], :]
        output_surface = pygame.transform.scale(
            pygame.surfarray.make_surface(output.transpose(1, 0, 2)),
            (self.surfw, self.surfh))
        self.window.fill(BORDER_COLOR, (
            self.surfw + 2 * self.padding - self.border_width,
            paney - self.border_width,
            self.surfw + 2 * self.border_width,
            self.surfh + 2 * self.border_width))
        self.window.blit(output_surface, (self.surfw + 2 * self.padding, paney))

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
                    f" rgb{self.colors[self.hovered]}",
                True, WHITE, BACKGROUND_COLOR)
            self.window.blit(surface, (
                self.width - surface.get_width() - self.padding,
                footery + self.square_size - surface.get_height()))

        pygame.display.flip()

    def get_hovered_color(self) -> tuple[int, int, int]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return self.window.get_at((mouse_x, mouse_y))[:3]

    def draw_hovered_color(self, flip=False):
        self.window.fill(self.get_hovered_color(), (
            self.padding + self.square_padding + self.square_size,
            self.height_sources + self.height_panes + 3 * self.padding,
            self.square_size,
            self.square_size))
        if flip:
            pygame.display.flip()

    def get_source(self, x: int, y: int) -> tuple[int, int] | None:
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
        x *= self.w / self.surfw
        y *= self.h / self.surfh
        if int(y) >= self.h or int(x) >= self.w:
            return None
        source = (
            self.mapping[int(y), int(x), 1],
            self.mapping[int(y), int(x), 0])
        if source in self.sources:
            return source
        return None

    def export(self):
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
            output_all = ask_export_format()
        array = numpy.zeros((*self.mapping.shape[:2], 4), dtype=numpy.uint8)
        for source in self.sources:
            if not output_all\
                and self.colors[source] == self.default_colors[source]:
                continue
            array[source[0], source[1]] = (*self.colors[source], 255)
        PIL.Image.fromarray(array).save(path)
        print(f"Exported to {path}")

    def over_buffer(self, x: int, y: int) -> bool:
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
    parser.add_argument("bitmap_path",
        type=str, default=None, nargs="?",
        help="path to image file")
    parser.add_argument("-m", "--max-sources-display",
        type=int, default=264,
        help="maximum number of sources to display")
    parser.add_argument("-w", "--width",
        type=int, default=1600,
        help="window width")
    args = parser.parse_args()
    window = Window(
        args.width,
        args.ckpt_path,
        args.bitmap_path,
        args.max_sources_display)
    with window:
        try:
            window.draw()
            while window.update():
                time.sleep(.001)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()