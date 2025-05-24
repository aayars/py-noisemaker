import os
import random

from PIL import Image, ImageDraw, ImageFont

import click
import qrcode
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer
from qrcode.image.styledpil import StyledPilImage

import textwrap


def mood_text(input_filename, text, font='LiberationSans-Bold', font_size=42, fill=None, rect=True, wrap_width=42, bottom=False, right=False, invert=False):
    if fill is None:
        if invert:
            fill = (0, 0, 0, 0)
        else:
            fill = (255, 255, 255, 255)

    image = Image.open(input_filename).convert('RGB')

    input_width, input_height = image.size

    font_path = os.path.join(os.path.expanduser('~'), '.noisemaker', 'fonts', '{}.ttf'.format(font))

    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(image, 'RGBA')

    padding = 6

    lines = textwrap.wrap(text, width=wrap_width)

    text_height = sum(draw.textsize(line, font=font)[1] + padding for line in lines)

    text_y = input_height - text_height

    if bottom:
        text_y -= int(padding * .5)

    else:
        text_y /= 2

    if invert:
        shadow_color = (255, 255, 255, 128)
    else:
        shadow_color = (0, 0, 0, 128)

    if rect:
        draw.rectangle(((0, text_y - padding), (input_width, text_y + text_height + padding)), fill=shadow_color)

    for i, line in enumerate(textwrap.wrap(text, width=wrap_width)):
        line_w, line_h = draw.textsize(line, font=font)

        text_x = input_width - line_w
        if right:
            text_x -= padding + 4
        else:
            text_x /= 2

        draw.text((text_x + 1, text_y + 1), line, font=font, fill=shadow_color)
        draw.text((text_x, text_y), line, font=font, fill=fill)

        text_y += line_h + padding

    image.save(input_filename)


def mood_qr(input_filename, data, fill, rect, bottom, right, background):
    image = Image.open(input_filename).convert('RGB')
    input_width, input_height = image.size

    qr_obj = qrcode.QRCode(
        version=None,
        error_correction=qrcode.ERROR_CORRECT_M,
        box_size=10,
        border=1
    )
    qr_obj.add_data(data)
    qr_obj.make(fit=True)

    qr_img = qr_obj.make_image(fill_color=fill[:3], back_color='black').convert('RGBA')
    max_qr_size = 64
    qr_width, qr_height = qr_img.size

    scale = min(max_qr_size / qr_width, max_qr_size / qr_height, 1)
    if scale < 1:
        new_size = (int(qr_width * scale), int(qr_height * scale))
        qr_img = qr_img.resize(new_size, Image.LANCZOS)
        qr_width, qr_height = new_size

    padding = 1
    y = input_height - qr_height - padding if bottom else (input_height - qr_height) // 2
    x = input_width - qr_width - padding - 1 if right else padding

    draw = ImageDraw.Draw(image, 'RGBA')
    if rect:
        draw.rectangle(
            ((x - padding, y - padding), (x + qr_width + padding, y + qr_height + padding)),
            fill=(0, 0, 0, 128)
        )

    image.paste(qr_img, (x, y), qr_img)
    image.save(input_filename)


@click.command()
@click.option('--filename', type=click.Path(dir_okay=False), required=True)
@click.option('--text', type=str, required=True)
@click.option('--font', type=str, default='LiberationSans-Bold')
@click.option('--font-size', type=int, default=42)
@click.option('--color', is_flag=True)
@click.option('--no-rect', is_flag=True)
@click.option('--wrap-width', type=int, default=42)
@click.option('--bottom', is_flag=True)
@click.option('--right', is_flag=True)
@click.option('--invert', is_flag=True)
@click.option('--qr', is_flag=True, help='Output a QR code instead of text')
def main(filename, text, font, font_size, color, no_rect, wrap_width, bottom, right, invert, qr):
    if color:
        if invert:
            fill = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255), 255)
        else:
            fill = (random.randint(0, 128), random.randint(0, 128), random.randint(0, 128), 255)

    else:
        if invert:
            fill = (255, 255, 255, 255)
        else:
            fill = (0, 0, 0, 255)

    if qr:
        # QR defaults to black modules unless a color fill was requested
        background = "black" if invert else "white"
        mood_qr(filename, text, fill, not no_rect, bottom, right, background)
    else:
        mood_text(filename, text, font, font_size, fill, not no_rect, wrap_width, bottom, right, invert)
