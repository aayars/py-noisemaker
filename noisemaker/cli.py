"""Common CLI boilerplate for Noisemaker"""

import click

from noisemaker.constants import (
    ColorSpace,
    DistanceMetric,
    InterpolationType,
    OctaveBlending,
    PointDistribution,
    ValueDistribution,
    ValueMask,
    VoronoiDiagramType,
    WormBehavior
)

from noisemaker.palettes import PALETTES as palettes

import noisemaker.masks as masks

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def show_values(enum_class):
    out = []

    for member in enum_class:
        out.append(f"{member.value}={member.name}")

    return f"({', '.join(out)})"


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}



def validate_more_than_one(allow_none=False):
    """
    """

    def validate(ctx, param, value):
        is_valid = False

        if value is None:
            is_valid = allow_none

        elif value > 1:
            is_valid = True

        if not is_valid:
            raise click.BadParameter("invalid choice: {0}. (choose a value greater than 1)".format(value))

        return value

    return validate


def validate_enum(cls):
    """
    """

    def validate(ctx, param, value):
        if value is not None and value not in [m.value for m in cls]:
            raise click.BadParameter("invalid choice: {0}. (choose from {1})".format(value, ", ".join(["{0} ({1})".format(m.value, m.name) for m in cls])))

        return value

    return validate


def bool_option(attr, **attrs):
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", attrs.get("default", False))

    return option(attr, **attrs)


def float_option(attr, **attrs):
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option(attr, **attrs)


def int_option(attr, **attrs):
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option(attr, **attrs)


def str_option(attr, **attrs):
    attrs.setdefault("type", str)
    attrs.setdefault("default", None)

    return option(attr, **attrs)


def multi_str_option(attr, **attrs):
    return str_option(attr, multiple=True, **attrs)


def option(*param_decls, **attrs):
    """ Add a Click option. """

    if "help" not in attrs:
        attrs["help"] = ""

    def decorator(f):
        if isinstance(attrs.get("type"), click.IntRange):
            r = attrs["type"]

            attrs["help"] += "  [range: {0}-{1}]".format(r.min, r.max)

        if attrs.get("default") not in (None, False, 0):
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def width_option(default=1024, **attrs):
    attrs.setdefault("help", "Output width, in pixels")

    return int_option("--width", default=default, **attrs)


def height_option(default=1024, **attrs):
    attrs.setdefault("help", "Output height, in pixels")

    return int_option("--height", default=default, **attrs)


def time_option(**attrs):
    attrs.setdefault("help", "Time value for Z axis (simplex only)")

    return float_option("--time", default=0.0, **attrs)


def seed_option(**attrs):
    attrs.setdefault("help", "Random seed. Might not affect all things.")

    return int_option("--seed", default=None, **attrs)


def filename_option(default=None, **attrs):
    attrs.setdefault("help", "Filename for image output (should end with .png or .jpg)")

    return str_option("--filename", type=click.Path(dir_okay=False), default=default or "noise.png", **attrs)


def input_dir_option(**attrs):
    attrs.setdefault("help", "Input directory containing .jpg and/or .png images")

    return str_option("--input-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), **attrs)
