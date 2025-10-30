from __future__ import annotations

"""Font loader for Noisemaker. Creates glyph atlases from TrueType fonts."""

import os

from PIL import Image, ImageDraw, ImageFont

import noisemaker.rng as random
from noisemaker.util import get_noisemaker_dir


def load_fonts() -> list[str]:
    """Find all TrueType fonts in the ~/.noisemaker/fonts directory.

    Returns:
        List of absolute paths to .ttf font files, or empty list if fonts directory doesn't exist.
    """

    fonts_dir = os.path.join(get_noisemaker_dir(), "fonts")

    if not os.path.exists(fonts_dir):
        return []

    return [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]


def load_glyphs(shape: list[int]) -> list[list[list[list[float]]]]:
    """Generate a list of ASCII character glyphs sorted from darkest to brightest.

    Renders printable ASCII characters (32-126) using a randomly selected font,
    then sorts them by brightness for use in value-based text rendering.

    RNG: One call to :func:`random.randint` to select font.

    Args:
        shape: Glyph dimensions as [height, width].

    Returns:
        List of glyphs, where each glyph is [y][x][channel] with normalized [0.0, 1.0] values.
        Sorted from darkest (space) to brightest (dense characters).
    """

    fonts = load_fonts()

    if not fonts:
        return []

    font_name = fonts[random.randint(0, len(fonts) - 1)]

    font = ImageFont.truetype(font_name, int(max(shape[0], shape[1]) * 0.9))

    glyphs = []
    totals = []

    for i in range(32, 127):
        total = 0

        glyph = []
        glyphs.append(glyph)

        image = Image.new("RGB", (shape[1], shape[0]))

        ImageDraw.Draw(image).text((0, 0), chr(i), font=font)

        for y in range(shape[0]):
            row = []
            glyph.append(row)

            for x in range(shape[1]):
                value = image.getpixel((x, y))[0] / 255

                row.append([value])
                total += value

        totals.append(total)

    return [g for total, g in sorted(zip(totals, glyphs))]
