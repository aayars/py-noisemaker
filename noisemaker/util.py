"""Utility functions for Noisemaker."""

from __future__ import annotations

import json
import os
import subprocess
from enum import Enum
from typing import Any

import tensorflow as tf
from loguru import logger as default_logger
from PIL import Image

from noisemaker.constants import ColorSpace


def save(tensor: tf.Tensor, name: str = "noise.png") -> None:
    """
    Save an image Tensor to a file.

    Args:
        tensor: Image tensor to save
        name: Filename, ending with .png or .jpg

    Returns:
        None
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


def load(filename: str, channels: int | None = None) -> tf.Tensor:
    """
    Load a .png or .jpg by filename.

    Args:
        filename: Path to the image file
        channels: Optional number of channels to force

    Returns:
        Loaded image tensor
    """

    with open(filename, "rb") as fh:
        if filename.lower().endswith(".png"):
            return tf.image.decode_png(fh.read(), channels=channels)

        elif filename.lower().endswith((".jpg", ".jpeg")):
            return tf.image.decode_jpeg(fh.read(), channels=channels)


def magick(pattern: str, name: str) -> Any:
    """
    Create a GIF from frames using ``ffmpeg``.

    Args:
        pattern: Frame filename pattern (e.g., ``/tmp/dir/*png`` or ``/tmp/dir/%04d.png``)
        name: Output filename

    Returns:
        Result of subprocess call
    """
    # Convert glob pattern to directory and ffmpeg pattern
    # If pattern is like "/tmp/dir/*png", convert to "/tmp/dir/%04d.png"
    import os
    import tempfile
    
    if "*" in pattern:
        directory = os.path.dirname(pattern)
        input_pattern = os.path.join(directory, "%04d.png")
    else:
        input_pattern = pattern
    
    # Use ffmpeg to create GIF with good quality palette
    # First generate a palette
    palette_file = os.path.join(tempfile.gettempdir(), "noisemaker_palette.png")
    
    palette_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "20",
        "-i",
        input_pattern,
        "-vf",
        "scale=flags=lanczos,palettegen",
        palette_file,
    ]
    
    check_call(palette_cmd, quiet=True)
    
    # Then use the palette to create the GIF
    gif_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "20",
        "-i",
        input_pattern,
        "-i",
        palette_file,
        "-lavfi",
        "scale=flags=lanczos[x];[x][1:v]paletteuse",
        name,
    ]
    
    return check_call(gif_cmd)


def watermark(text: str, filename: str) -> Any:
    """
    Annotate an image.

    Args:
        text: Text to add to the image
        filename: Image filename to annotate

    Returns:
        Result of subprocess call
    """

    return check_call(
        ["mood", "--filename", filename, "--text", text, "--font", "Nunito-VariableFont_wght", "--font-size", "12", "--no-rect", "--bottom", "--right"]
    )


def check_call(command: list[str], quiet: bool = False) -> Any:
    """Execute a subprocess command."""
    try:
        subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

    except Exception as e:
        if not quiet:
            log_subprocess_error(command, e)

        raise


def log_subprocess_error(command: str | list[str], e: Exception) -> None:
    """Log subprocess execution errors with appropriate detail.

    Args:
        command: The command that failed.
        e: The exception that was raised.
    """
    if isinstance(e, subprocess.CalledProcessError):
        logger.error(f"{e}: {e.output.strip()}")

    else:
        logger.error(f"Command '{command}' failed to execute: {e}")


def get_noisemaker_dir() -> str:
    """Get the Noisemaker configuration directory path.

    Returns:
        Path to ~/.noisemaker or NOISEMAKER_DIR environment variable.
    """
    return os.environ.get("NOISEMAKER_DIR", os.path.join(os.path.expanduser("~"), ".noisemaker"))


def dumps(kwargs: dict[str, Any]) -> str:
    """Serialize kwargs to JSON, converting Enums to strings.

    Args:
        kwargs: Dictionary to serialize.

    Returns:
        Pretty-printed JSON string with sorted keys.
    """
    out = {}

    for k, v in kwargs.items():
        if isinstance(v, Enum):
            out[k] = str(v)
        else:
            out[k] = v

    return json.dumps(out, indent=4, sort_keys=True)


def shape_from_params(width: int, height: int, color_space: ColorSpace, with_alpha: bool) -> list[int]:
    """Construct a shape array from image parameters.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color_space: Color space (grayscale or color).
        with_alpha: Include alpha channel.

    Returns:
        Shape list [height, width, channels].
    """
    if color_space == ColorSpace.grayscale:
        shape = [height, width, 1]

    else:
        shape = [height, width, 3]

    if with_alpha:
        shape[2] += 1

    return shape


def shape_from_file(filename: str) -> list[int]:
    """Extract image dimensions from a file using PIL.

    Uses PIL to avoid adding operations to the TensorFlow computation graph.

    Args:
        filename: Path to image file.

    Returns:
        Shape list [height, width, channels].
    """

    image = Image.open(filename)

    input_width, input_height = image.size

    return [input_height, input_width, len(image.getbands())]


def from_srgb(srgb: tf.Tensor) -> tf.Tensor:
    """Convert an sRGB image tensor to Linear RGB color space.

    Args:
        srgb: Image tensor in sRGB color space.

    Returns:
        Image tensor in Linear RGB color space.
    """
    condition = tf.less(srgb, 0.04045)
    linear_rgb = tf.where(condition, srgb / 12.92, tf.pow((srgb + 0.055) / 1.055, 2.4))
    return linear_rgb


def from_linear_rgb(linear_rgb: tf.Tensor) -> tf.Tensor:
    """Convert a Linear RGB image tensor to sRGB color space.

    Args:
        linear_rgb: Image tensor in Linear RGB color space.

    Returns:
        Image tensor in sRGB color space.
    """
    condition = tf.less(linear_rgb, 0.0031308)
    srgb = tf.where(condition, linear_rgb * 12.92, 1.055 * tf.pow(linear_rgb, 1 / 2.4) - 0.055)
    return srgb


_LOGS_DIR = os.path.join(get_noisemaker_dir(), "logs")

os.makedirs(_LOGS_DIR, exist_ok=True)

logger = default_logger
# logger.remove(0)  # Remove loguru's default STDERR log handler
logger.add(os.path.join(_LOGS_DIR, "noisemaker.log"), retention="7 days")
