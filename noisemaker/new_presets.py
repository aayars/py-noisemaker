import functools
import random

from noisemaker.composer import Effect, Preset
from noisemaker.constants import (
    InterpolationType as interp,
    ValueMask as mask,
)

PRESETS = {
    "aberration": {
        "post": lambda settings: [
            Effect("aberration", displacement=.025 + random.random() * .0125)
        ]
    },

    "be-kind-rewind": {
        "post": lambda settings: [
            Effect("vhs"),
            Preset("crt"),
        ]
    },

    "bloom": {
        "post": lambda settings: [
            Effect("bloom", alpha=.25 + random.random() * .125)
        ]
    },

    "carpet": {
        "post": lambda settings: [
            Effect("worms", alpha=.4, density=250, duration=.75, stride=.5, stride_deviation=.25),
            Effect("grime"),
        ]
    },

    "clouds": {
        "post": lambda settings: [
            Effect("clouds"),
            Preset("bloom"),
            Preset("dither"),
        ]
    },

    "convolution-feedback": {
        "post": lambda settings: [
            Effect("conv_feedback",
                   alpha=.5 * random.random() * .25,
                   iterations=random.randint(250, 500)),
        ]
    },

    "corrupt": {
        "post": lambda settings: [
            Effect("warp",
                   displacement=.025 + random.random() * .1,
                   freq=[random.randint(2, 4), random.randint(1, 3)],
                   octaves=random.randint(2, 4),
                   spline_order=interp.constant),
        ]
    },

    "crt": {
        "extends": ["scanline-error", "snow"],

        "post": lambda settings: [
            Effect("crt"),
        ]
    },

    "degauss": {
        "post": lambda settings: [
            Effect("degauss", displacement=.0625 + random.random() * .03125),
            Preset("crt"),
        ]
    },

    "density-map": {
        "post": lambda settings: [
            Effect("density_map"),
            Effect("convolve", kernel=mask.conv2d_invert),
            Preset("dither"),
        ]
    },

    "desaturate": {
        "post": lambda settings: [
            Effect("adjust_saturation", amount=.333 + random.random() * .16667)
        ]
    },

    "distressed": {
        "extends": ["dither", "filthy"],
        "post": lambda settings: [
            Preset("desaturate"),
        ]
    },

    "dither": {
        "post": lambda settings: [
            Effect("dither", alpha=.125 + random.random() * .06125),
        ]
    },

    "filthy": {
        "post": lambda settings: [
            Effect("grime"),
            Effect("scratches"),
            Effect("stray_hair"),
        ]
    },

    "scanline-error": {
        "post": lambda settings: [
            Effect("scanline_error"),
        ]
    },

    "snow": {
        "post": lambda settings: [
            Effect("snow", alpha=.333 + random.random() * .16667),
        ]
    },
}

Preset = functools.partial(Preset, presets=PRESETS)
