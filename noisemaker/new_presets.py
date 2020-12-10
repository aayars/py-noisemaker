from noisemaker.composer import Effect

import random

PRESETS = {
    "aberration": {
        "post": lambda settings: [
            Effect("aberration", displacement=.025 + random.random() * .0125)
        ]
    },

    "be-kind-rewind": {
        "post": lambda settings: [
            Effect("vhs"),
            Effect("crt"),
        ]
    },

    "crt": {
        "extends": ["scanline-error", "snow"],

        "post": lambda settings: [
            Effect("crt"),
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
