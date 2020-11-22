"""Utility functions for Noisemaker."""

import os
import subprocess

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
        return check_call(['magick'] + common_params)

    except FileNotFoundError:
        return check_call(['convert'] + common_params)


def watermark(text, filename):
    """
    Annotate an image.

    :param text:
    :param filename:
    """

    return check_call(['mood',
                       '--filename', filename,
                       '--text', text,
                       '--font', 'LiberationSans-Bold',
                       '--font-size', '12',
                       '--no-rect',
                       '--bottom',
                       '--right'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_call(command, **kwargs):
    try:
        result = subprocess.check_call(command, **kwargs)

    except subprocess.CalledProcessError as e:
        logger.error(e)
        raise

    except Exception as e:
        logger.error(f"Command '{command}' failed to execute: {e}")
        raise



def get_noisemaker_dir():
    return os.environ.get('NOISEMAKER_DIR', os.path.join(os.path.expanduser("~"), '.noisemaker'))


_LOGS_DIR = os.path.join(get_noisemaker_dir(), 'logs')

os.makedirs(_LOGS_DIR, exist_ok=True)

logger = default_logger
logger.remove(0)  # Remove loguru's default STDERR log handler
logger.add(os.path.join(_LOGS_DIR, "noisemaker.log"), retention="7 days")
