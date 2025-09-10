"""Font loader for Noisemaker. Creates glyph atlases from TrueType fonts."""

import os
import noisemaker.rng as random

from PIL import Image, ImageDraw, ImageFont

from noisemaker.util import get_noisemaker_dir


def load_fonts():
    """
    Finds all TrueType fonts in your ~/.noisemaker/fonts directory.
    """

    fonts_dir = os.path.join(get_noisemaker_dir(), "fonts")

    if not os.path.exists(fonts_dir):
        return []

    return [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith(".ttf")]


def load_glyphs(shape):
    """
    Return a list of glyphs, sorted from darkest to brightest.

    :param list[int] shape:
    """

    fonts = load_fonts()

    if not fonts:
        return []

    font_name = fonts[random.randint(0, len(fonts) - 1)]

    font = ImageFont.truetype(font_name, int(max(shape[0], shape[1]) * .9))

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
