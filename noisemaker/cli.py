from __future__ import annotations

"""Common CLI boilerplate for Noisemaker"""

from enum import Enum
from typing import Any, Callable

import click

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def show_values(enum_class: type[Enum]) -> str:
    """Format enum values for CLI help text.

    Args:
        enum_class: Enum class to format.

    Returns:
        Formatted string like "(0=value_a, 1=value_b, 2=value_c)".
    """
    out = []

    for member in enum_class:
        out.append(f"{member.value}={member.name}")

    return f"({', '.join(out)})"


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 160}


def validate_more_than_one(allow_none: bool = False) -> Callable:
    """Create a Click validator that requires values greater than 1.

    Args:
        allow_none: If True, None values are allowed.

    Returns:
        Validator function for Click options.

    Raises:
        click.BadParameter: If value is not greater than 1 (and not None when allowed).
    """

    def validate(ctx, param, value):
        is_valid = False

        if value is None:
            is_valid = allow_none

        elif value > 1:
            is_valid = True

        if not is_valid:
            raise click.BadParameter(f"invalid choice: {value}. (choose a value greater than 1)")

        return value

    return validate


def validate_enum(cls: type[Enum]) -> Callable:
    """Create a Click validator that enforces enum membership.

    Args:
        cls: Enum class to validate against.

    Returns:
        Validator function for Click options.

    Raises:
        click.BadParameter: If value is not a valid enum member.
    """

    def validate(ctx, param, value):
        if value is not None and value not in [m.value for m in cls]:
            raise click.BadParameter("invalid choice: {0}. (choose from {1})".format(value, ", ".join([f"{m.value} ({m.name})" for m in cls])))

        return value

    return validate


def bool_option(attr: str, **attrs: Any) -> Callable:
    """Create a boolean flag option (defaults to False).

    Args:
        attr: Option name (e.g., "--verbose").
        **attrs: Additional Click option attributes.

    Returns:
        Decorator for the option.
    """
    attrs.setdefault("is_flag", True)
    attrs.setdefault("default", attrs.get("default", False))

    return option(attr, **attrs)


def float_option(attr: str, **attrs: Any) -> Callable:
    """Create a float option (defaults to 0.0).

    Args:
        attr: Option name (e.g., "--time").
        **attrs: Additional Click option attributes.

    Returns:
        Decorator for the option.
    """
    attrs.setdefault("type", float)
    attrs.setdefault("default", 0.0)

    return option(attr, **attrs)


def int_option(attr: str, **attrs: Any) -> Callable:
    """Create an integer option (defaults to 0).

    Args:
        attr: Option name (e.g., "--width").
        **attrs: Additional Click option attributes.

    Returns:
        Decorator for the option.
    """
    attrs.setdefault("type", int)
    attrs.setdefault("default", 0)

    return option(attr, **attrs)


def str_option(attr: str, **attrs: Any) -> Callable:
    """Create a string option (defaults to None).

    Args:
        attr: Option name (e.g., "--filename").
        **attrs: Additional Click option attributes.

    Returns:
        Decorator for the option.
    """
    attrs.setdefault("type", str)
    attrs.setdefault("default", None)

    return option(attr, **attrs)


def multi_str_option(attr: str, **attrs: Any) -> Callable:
    """Create a multi-value string option (accepts multiple values).

    Args:
        attr: Option name (e.g., "--input").
        **attrs: Additional Click option attributes.

    Returns:
        Decorator for the option.
    """
    return str_option(attr, multiple=True, **attrs)


def option(*param_decls: Any, **attrs: Any) -> Callable:
    """Create a Click option decorator with automatic help text formatting.

    Adds range and default value information to help text automatically.

    Args:
        *param_decls: Click parameter declarations (e.g., "--width", "-w").
        **attrs: Click option attributes (type, default, help, etc.).

    Returns:
        Decorator function for adding the option to a Click command.
    """

    if "help" not in attrs:
        attrs["help"] = ""

    def decorator(f):
        if isinstance(attrs.get("type"), click.IntRange):
            r = attrs["type"]

            attrs["help"] += f"  [range: {r.min}-{r.max}]"

        if attrs.get("default") not in (None, False, 0):
            attrs["help"] += "  [default: {0}]".format(attrs["default"])

        return click.option(*param_decls, **attrs)(f)

    return decorator


def width_option(default: int = 1024, **attrs: Any) -> Callable:
    attrs.setdefault("help", "Output width, in pixels")

    return int_option("--width", default=default, **attrs)


def height_option(default: int = 1024, **attrs: Any) -> Callable:
    attrs.setdefault("help", "Output height, in pixels")

    return int_option("--height", default=default, **attrs)


def time_option(**attrs: Any) -> Callable:
    attrs.setdefault("help", "Time value for Z axis (simplex only)")

    return float_option("--time", default=0.0, **attrs)


def seed_option(**attrs: Any) -> Callable:
    attrs.setdefault("help", "Random seed. Might not affect all things.")

    return int_option("--seed", default=None, **attrs)


def filename_option(default: str | None = None, **attrs: Any) -> Callable:
    attrs.setdefault("help", "Filename for image output (should end with .png or .jpg)")

    return str_option("--filename", type=click.Path(dir_okay=False), default=default or "noise.png", **attrs)


def input_dir_option(**attrs: Any) -> Callable:
    attrs.setdefault("help", "Input directory containing .jpg and/or .png images")

    return str_option("--input-dir", type=click.Path(exists=True, file_okay=False, resolve_path=True), **attrs)
