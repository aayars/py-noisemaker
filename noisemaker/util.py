"""Utility functions for Noisemaker."""

from enum import Enum

import json
import os
import subprocess

from noisemaker.constants import ColorSpace

from PIL import Image
from loguru import logger as default_logger

import tensorflow as tf


def save(tensor, name="noise.png"):
    """
    Save an image Tensor to a file.

    :param Tensor tensor: Image tensor
    :param str name: Filename, ending with .png or .jpg
    :return: None
    """

    tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)

    if name.lower().endswith(".png"):
        data = tf.image.encode_png(tensor).numpy()

    elif name.lower().endswith((".jpg", ".jpeg")):
        data = tf.image.encode_jpeg(tensor).numpy()

    else:
        raise ValueError("Filename should end with .png or .jpg")

    with open(name, "wb") as fh:
        fh.write(data)


def load(filename, channels=None):
    """
    Load a .png or .jpg by filename.

    :param str filename:
    :return: Tensor
    """

    with open(filename, "rb") as fh:
        if filename.lower().endswith(".png"):
            return tf.image.decode_png(fh.read(), channels=channels)

        elif filename.lower().endswith((".jpg", ".jpeg")):
            return tf.image.decode_jpeg(fh.read(), channels=channels)


def magick(glob, name):
    """
    Shell out to ImageMagick's "convert" (im6) or "magick" (im7) commands for GIF composition, depending on what's available.

    :param str glob: Frame filename glob pattern
    :param str name: Filename
    """

    common_params = ['-delay', '5', glob, name]

    try:
        command = 'magick'
        return check_call([command] + common_params, quiet=True)

    except FileNotFoundError:
        command = 'convert'
        return check_call([command] + common_params)

    except Exception as e:
        log_subprocess_error(command, e)  # Try to only log non-pathological errors from `magick`


def watermark(text, filename):
    """
    Annotate an image.

    :param text:
    :param filename:
    """

    return check_call(['mood',
                       '--filename', filename,
                       '--text', text,
                       '--font', 'Nunito-VariableFont_wght',
                       '--font-size', '12',
                       '--no-rect',
                       '--bottom',
                       '--right'])


def check_call(command, quiet=False):
    try:
        subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

    except Exception as e:
        if not quiet:
            log_subprocess_error(command, e)

        raise


def log_subprocess_error(command, e):
    if isinstance(e, subprocess.CalledProcessError):
        logger.error(f"{e}: {e.output.strip()}")

    else:
        logger.error(f"Command '{command}' failed to execute: {e}")


def get_noisemaker_dir():
    return os.environ.get('NOISEMAKER_DIR', os.path.join(os.path.expanduser("~"), '.noisemaker'))


def dumps(kwargs):
    out = {}

    for k, v in kwargs.items():
        if isinstance(v, Enum):
            out[k] = str(v)
        else:
            out[k] = v

    return json.dumps(out, indent=4, sort_keys=True)


def shape_from_params(width, height, color_space, with_alpha):
    if color_space == ColorSpace.grayscale:
        shape = [height, width, 1]

    else:
        shape = [height, width, 3]

    if with_alpha:
        shape[2] += 1

    return shape


def shape_from_file(filename):
    """
    Get image dimensions from a file, using PIL, to avoid adding to the TensorFlow graph.
    """

    image = Image.open(filename)

    input_width, input_height = image.size

    return [input_height, input_width, len(image.getbands())]


def from_srgb(srgb):
    """Converts an sRGB image to Linear RGB."""
    condition = tf.less(srgb, 0.04045)
    linear_rgb = tf.where(condition, srgb / 12.92, tf.pow((srgb + 0.055) / 1.055, 2.4))
    return linear_rgb


def from_linear_rgb(linear_rgb):
    """Converts a Linear RGB image to sRGB."""
    condition = tf.less(linear_rgb, 0.0031308)
    srgb = tf.where(condition, linear_rgb * 12.92, 1.055 * tf.pow(linear_rgb, 1/2.4) - 0.055)
    return srgb


_LOGS_DIR = os.path.join(get_noisemaker_dir(), 'logs')

os.makedirs(_LOGS_DIR, exist_ok=True)

logger = default_logger
# logger.remove(0)  # Remove loguru's default STDERR log handler
logger.add(os.path.join(_LOGS_DIR, "noisemaker.log"), retention="7 days")
