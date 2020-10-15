"""Presets library for artmaker/artmangler scripts.

Presets may contain any keyword arg accepted by :func:`~noisemaker.effects.post_process()` or :func:`~noisemaker.recipes.post_process()`.
"""

from collections import deque
from enum import Enum, EnumMeta

import random

from noisemaker.constants import (
    DistanceFunction as df,
    PointDistribution as pd,
    ValueDistribution,
    ValueMask as vm,
    VoronoiDiagramType as voronoi
)

import noisemaker.generators as generators
import noisemaker.masks as masks


# Baked presets go here
EFFECTS_PRESETS = {}

PRESETS = {}

# Stashed random values
_STASH = {}

# Use a lambda to permit re-eval with new seed
_EFFECTS_PRESETS = lambda: {  # noqa: E731
    "aberration": lambda: {
        "with_aberration": .015 + random.random() * .015,
    },

    "be-kind-rewind": lambda: extend("crt", {
        "with_vhs": True,
    }),

    "bloom": lambda: {
        "with_bloom": .075 + random.random() * .075,
    },

    "carpet": lambda: {
        "with_grime": True,
        "with_worms": 4,
        "worms_alpha": .5,
        "worms_density": 250,
        "worms_duration": .75,
        "worms_stride": .5,
        "worms_stride_deviation": .25,
    },

    "clouds": lambda: {
        "with_clouds": True,
        "with_bloom": .25,
        "with_dither": .05,
    },

    "convolution-feedback": lambda: {
        "conv_feedback_alpha": .5,
        "with_conv_feedback": 500,
    },

    "corrupt": lambda: {
        "warp_freq": [random.randint(2, 4), random.randint(1, 3)],
        "warp_octaves": random.randint(2, 4),
        "warp_range": .025 + random.random() * .1,
        "warp_interp": 0,
    },

    "crt": lambda: extend("scanline-error", "snow", {
        "with_crt": True,
    }),

    "density-map": lambda: extend("invert", "dither", {
        "with_density_map": True,
    }),

    "distressed": lambda: extend("dither", "filthy", "scuff", {
        "post_saturation": .25 + random.random() * .125,
    }),

    "dither": lambda: {
        "with_dither": .125 + random.random() * .125,
    },

    "erosion-worms": lambda: {
        "erosion_worms_alpha": .25 + random.random() * .75,
        "erosion_worms_contraction": .5 + random.random() * .5,
        "erosion_worms_density": random.randint(25, 100),
        "erosion_worms_iterations": random.randint(25, 100),
        "with_erosion_worms": True,
    },

    "extract-derivative": lambda: {
        "deriv": random.randint(1, 3),
    },

    "falsetto": lambda: {
        "with_false_color": True
    },

    "filthy": lambda: {
        "with_grime": True,
        "with_scratches": random.randint(0, 1),
        "with_stray_hair": True,
    },

    "funhouse": lambda: {
        "warp_signed_range": False,
        "warp_interp": 3,
        "warp_freq": [random.randint(2, 4), random.randint(1, 4)],
        "warp_octaves": random.randint(1, 4),
        "warp_range": .25 + random.random() * .5,
    },

    "glitchin-out": lambda: extend("bloom", "corrupt", "crt", {
        "with_glitch": True,
        "with_ticker": (random.random() > .75),
    }),

    "glowing-edges": lambda: {
        "with_glowing_edges": 1.0,
    },

    "glyph-map": lambda: {
        "glyph_map_colorize": random.randint(0, 1),
        "glyph_map_zoom": random.randint(1, 3),
        "with_glyph_map": random_member(set(vm.procedural_members()).intersection(masks.square_masks())),
    },

    "grayscale": lambda: {
        "post_saturation": 0,
    },

    "invert": lambda: {
        "with_convolve": ["invert"],
    },

    "lens": lambda: extend("aberration", "vaseline", "tint", "snow", {
        "with_vignette": .125 + random.random() * .125,
    }),

    "lens-warp": lambda: {
        "speed": .05,
        "with_lens_warp": .25 + random.random() * .25,
    },

    "light-leak": lambda: extend("bloom", "vignette-bright", {
        "with_light_leak": .333 + random.random() * .333,
    }),

    "lowpoly": lambda: {
        "with_lowpoly": True,
    },

    "maybe-invert": lambda: {
        "with_convolve": [] if random.randint(0, 1) else ["invert"],
    },

    "mosaic": lambda: extend("bloom", "voronoi", {
        "voronoi_alpha": .75 + random.random() * .25,
        "with_voronoi": voronoi.range_regions,
    }),

    "nebula": lambda: {
        "with_nebula": True,
    },

    "noirmaker": lambda: extend("bloom", "dither", "vignette-dark", {
        "post_contrast": 5,
        "post_saturation": 0,
        "with_light_leak": .25 + random.random() * .25,
    }),

    "normals": lambda: {
        "with_normal_map": True,
    },

    "octave-warp": lambda: {
        "speed": .0125,
        "warp_freq": random.randint(2, 3),
        "warp_octaves": 3,
        "warp_range": 3.0 + random.random(),
    },

    "one-art-please": lambda: extend("dither", "light-leak", {
        "post_contrast": 1.25,
        "post_saturation": .75,
        "with_texture": True,
    }),

    "outline": lambda: {
        "with_outline": random.randint(1, 3),
    },

    "pixel-sort": lambda: {
        "with_sort": True,
    },

    "pixel-sort-angled": lambda: extend("pixel-sort", {
        "sort_angled": True,
    }),

    "pixel-sort-darkest": lambda: extend("pixel-sort", {
        "sort_darkest": True,
    }),

    "pixel-sort-angled-darkest": lambda: extend("pixel-sort-angled", "pixel-sort-darkest"),

    "posterize-outline": lambda: extend("outline", {
        "posterize_levels": random.randint(3, 7),
    }),

    "random-effect": lambda:
        preset(random_member([m for m in EFFECTS_PRESETS if m != "random-effect"])),

    "random-hue": lambda: {
        "post_hue_rotation": random.random()
    },

    "reflect-domain-warp": lambda: {
        "reflect_range": .5 + random.random() * 12.5,
    },

    "refract-domain-warp": lambda: {
        "refract_range": .125 + random.random() * 1.25,
    },

    "reindex": lambda: {
        "reindex_range": .125 + random.random() * 2.5,
    },

    "reverb": lambda: {
        "reverb_iterations": random.randint(1, 4),
        "with_reverb": random.randint(3, 6),
    },

    "ripples": lambda: {
        "ripple_freq": random.randint(2, 3),
        "ripple_kink": random.randint(3, 18),
        "ripple_range": .025 + random.random() * .1,
    },

    "scanline-error": lambda: {
        "with_scan_error": True,
    },

    "scuff": lambda: {
        "with_scratches": True,
    },

    "shadows": lambda: extend("vignette-dark", {
        "with_shadow": .5 + random.random() * .5,
    }),

    "shake-it-like": lambda: {
        "with_frame": True,
    },

    "simple-frame": lambda: {
        "with_simple_frame": True
    },

    "sketch": lambda: {
        "with_grime": True,
        "with_fibers": True,
        "with_sketch": True,
        "with_texture": True,
    },

    "snow": lambda: {
        "with_snow": .5 + random.random() * .5,
    },

    "sobel": lambda: extend("maybe-invert", {
        "with_sobel": random.randint(1, 3),
    }),

    "spatter": lambda: {
        "speed": .05,
        "with_spatter": True,
    },

    "spooky-ticker": lambda: {
        "with_ticker": True,
    },

    "subpixels": lambda: {
        "composite_zoom": random_member([2, 4, 8]),
        "with_composite": random_member(vm.rgb_members()),
    },

    "swerve-h": lambda: {
        "warp_freq": [random.randint(2, 5), 1],
        "warp_interp": 3,
        "warp_octaves": 1,
        "warp_range": .5 + random.random() * .5,
    },

    "swerve-v": lambda: {
        "warp_freq": [1, random.randint(2, 5)],
        "warp_interp": 3,
        "warp_octaves": 1,
        "warp_range": .5 + random.random() * .5,
    },

    "tensor-tone": lambda: {
        "glyph_map_colorize": random.randint(0, 1),
        "with_glyph_map": "halftone",
    },

    "tint": lambda: {
        "with_tint": .333 + random.random() * .333,
    },

    "vaseline": lambda: {
        "with_vaseline": .625 + random.random() * .25,
    },

    "vignette-bright": lambda: {
        "with_vignette": .333 + random.random() * .333,
        "vignette_brightness": 1,
    },

    "vignette-dark": lambda: {
        "with_vignette": .65 + random.random() * .35,
        "vignette_brightness": 0,
    },

    "voronoi": lambda: {
        "point_distrib": pd.random if random.randint(0, 1) else random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(4, 10),
        "voronoi_func": random_member(df.all()),
        "voronoi_inverse": random.randint(0, 1),
        "voronoi_nth": random.randint(0, 2),
        "with_voronoi": random_member([t for t in voronoi if t != voronoi.none]),
    },

    "voronoid": lambda: extend("voronoi", {
        "voronoi_refract": .25 + random.random() * .25,
        "with_voronoi": random_member([voronoi.range, voronoi.regions, voronoi.flow]),
    }),

    "vortex": lambda: {
        "vortex_range": random.randint(16, 48),
    },

    "wormhole": lambda: {
        "with_wormhole": True,
        "wormhole_stride": .025 + random.random() * .05,
        "wormhole_kink": .5 + random.random(),
    },

    "worms": lambda: {
        "with_worms": random.randint(1, 5),
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 500,
        "worms_duration": 1,
        "worms_kink": 2.5,
        "worms_stride": 2.5,
        "worms_stride_deviation": 2.5,
    },

}

_PRESETS = lambda: {  # noqa: E731
    "1969": lambda: extend("density-map", "distressed", "posterize-outline", "nerdvana", {
        "point_corners": True,
        "point_distrib": "circular",
        "point_freq": random.randint(3, 5) * 2,
        "reflect_range": 2,
        "rgb": True,
        "voronoi_alpha": .5 + random.random() * .5,
        "with_bloom": False,
        "with_reverb": False,
        "with_voronoi": voronoi.color_range,
    }),

    "1976": lambda: extend("dither", {
        "point_freq": 2,
        "post_saturation": .125 + random.random() * .0625,
        "voronoi_func": df.triangular,
        "with_voronoi": voronoi.color_regions,
    }),

    "1985": lambda: extend("spatter", {
        "freq": random.randint(15, 25),
        "reindex_range": .2 + random.random() * .1,
        "rgb": True,
        "spline_order": 0,
        "voronoi_func": 3,
        "voronoi_refract": .2 + random.random() * .1,
        "with_voronoi": voronoi.range,
    }),

    "2001": lambda: extend("aberration", "bloom", "invert", "value-mask", {
        "freq": 13 * random.randint(10, 20),
        "mask": "bank_ocr",
        "posterize_levels": 1,
        "vignette_brightness": 1,
        "with_vignette": 1,
    }),

    "2d-chess": lambda: extend("value-mask", {
        "corners": True,
        "freq": 8,
        "mask": "chess",
        "point_corners": True,
        "point_distrib": "square",
        "point_freq": 8,
        "spline_order": 0,
        "voronoi_alpha": 0.5 + random.random() * .5,
        "voronoi_nth": random.randint(0, 1) * random.randint(0, 63),
        "with_voronoi": voronoi.color_range if random.randint(0, 1) \
            else random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "acid": lambda: {
        "freq": random.randint(10, 15),
        "octaves": 8,
        "post_reindex_range": 1.25 + random.random() * 1.25,
        "rgb": True,
    },

    "acid-droplets": lambda: extend("bloom", "density-map", "multires-low", "random-hue", {
        "freq": random.randint(12, 18),
        "hue_range": 0,
        "mask": "sparse",
        "mask_static": True,
        "post_saturation": .25,
        "reflect_range": 3.75 + random.random() * 3.75,
        "ridges": random.randint(0, 1),
        "saturation": 1.5,
        "with_shadow": 1,
    }),

    "acid-grid": lambda: extend("bloom", "funhouse", "sobel", "voronoid", {
        "lattice_drift": random.randint(0, 1),
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": 4,
        "point_generations": 2,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_func": 1,
        "with_voronoi": voronoi.color_range,
    }),

    "acid-wash": lambda: extend("funhouse", "reverb", "symmetry", {
        "hue_range": 1,
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(6, 10),
        "post_ridges": True,
        "ridges": True,
        "saturation": .25,
        "voronoi_alpha": .333 + random.random() * .333,
        "warp_octaves": 8,
        "with_shadow": 1,
        "with_voronoi": voronoi.color_range,
    }),

    "activation-signal": lambda: extend("glitchin-out", "value-mask", {
        "freq": 4,
        "mask": "white_bear",
        "rgb": random.randint(0, 1),
        "spline_order": 0,
        "with_vhs": random.randint(0, 1),
    }),

    "aesthetic": lambda: extend("be-kind-rewind", "maybe-invert", {
        "corners": True,
        "deriv": random.randint(0, 1),
        "distrib": random_member(["column_index", "ones", "row_index"]),
        "freq": random.randint(3, 5) * 2,
        "mask": "chess",
        "spline_order": 0,
        "with_pre_spatter": True,
    }),

    "alien-terrain-multires": lambda: extend("bloom", "maybe-invert", "multires", {
        "deriv": 1,
        "deriv_alpha": .333 + random.random() * .333,
        "freq": random.randint(4, 8),
        "lattice_drift": 1,
        "post_saturation": .075 + random.random() * .075,
        "saturation": 2,
        "with_shadow": .75 + random.random() * .25,
    }),

    "alien-terrain-worms": lambda: extend("bloom", "dither", "erosion-worms", "multires-ridged", {
        "deriv": 1,
        "deriv_alpha": 0.25 + random.random() * .125,
        "erosion_worms_alpha": .025 + random.random() * .015,
        "erosion_worms_density": random.randint(150, 200),
        "erosion_worms_inverse": True,
        "erosion_worms_xy_blend": .42,
        "freq": random.randint(3, 5),
        "hue_rotation": .875,
        "hue_range": .25 + random.random() * .25,
        "point_freq": 10,
        "post_contrast": 1.25,
        "post_saturation": .25,
        "saturation": 2,
        "voronoi_alpha": 0.125 + random.random() * .125,
        "voronoi_refract": 0.125 + random.random() * .125,
        "with_shadow": .333,
        "with_voronoi": voronoi.flow,
    }),

    "alien-transmission": lambda: extend("glitchin-out", "sobel", "value-mask", {
        "mask": stash("alien-transmission-mask", random_member(vm.procedural_members())),
        # offset by i * .5 for glitched texture lookup
        "freq": [int(i * .5 + i * stash("alien-transmission-repeat", random.randint(20, 30)))
            for i in masks.mask_shape(stash("alien-transmission-mask"))[0:2]],
    }),

    "analog-glitch": lambda: extend("value-mask", {
        "mask": stash("analog-glitch-mask", random_member([vm.alphanum_hex, vm.lcd, vm.fat_lcd])),
        "deriv": 2,
        # offset by i * .5 for glitched texture lookup
        "freq": [i * .5 + i * stash("analog-glitch-repeat", random.randint(20, 30))
            for i in masks.mask_shape(stash("analog-glitch-mask"))[0:2]],
    }),

    "anticounterfeit": lambda: extend("dither", "invert", "wormhole", {
        "freq": 2,
        "point_freq": 1,
        "voronoi_refract": .5,
        "with_fibers": True,
        "with_voronoi": voronoi.flow,
        "with_watermark": True,
        "wormhole_kink": 6,
    }),

    "arcade-carpet": lambda: extend("basic", "dither", {
        "post_hue_rotation": -.125,
        "distrib": ValueDistribution.exp,
        "freq": random.randint(75, 125),
        "hue_range": 1,
        "mask": "sparser",
        "mask_static": True,
        "posterize_levels": 3,
        "post_contrast": 1.25,
        "warp_range": 0.025,
        "rgb": True,
    }),

    "are-you-human": lambda: extend("aberration", "density-map", "funhouse", "maybe-invert", "multires", "snow", "value-mask", {
        "freq": 15,
        "hue_range": random.random() * .25,
        "hue_rotation": random.random(),
        "mask": "truetype",
        "saturation": random.random() * .125,
    }),

    "aztec-waffles": lambda: extend("maybe-invert", "outline", {
        "freq": 7,
        "point_freq": random.randint(2, 4),
        "point_generations": 2,
        "point_distrib": "circular",
        "posterize_levels": random.randint(6, 18),
        "reflect_range": random.random(),
        "spline_order": 0,
        "voronoi_func": random.randint(2, 3),
        "voronoi_nth": random.randint(2, 4),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "bad-map": lambda: extend("splork", {
        "freq": random.randint(30, 50),
        "mask": "grid",
        "with_voronoi": voronoi.range,
    }),

    "basic": lambda: {
        "freq": random.randint(2, 4),
    },

    "basic-lowpoly": lambda: extend("basic", "lowpoly"),

    "basic-voronoi": lambda: extend("basic", "voronoi"),

    "basic-voronoi-refract": lambda: extend("basic", {
        "hue-range": .25 + random.random() * .5,
        "voronoi_refract": .5 + random.random() * .5,
        "with_voronoi": voronoi.range,
    }),

    "band-together": lambda: {
        "freq": random.randint(6, 12),
        "reindex_range": random.randint(8, 12),
        "warp_range": .5,
        "warp_octaves": 8,
        "warp_freq": 2,
        "with_shadow": .25 + random.random() * .25,
    },

    "benny-lava": lambda: extend("distressed", {
        "distrib": "column_index",
        "posterize_levels": 1,
        "warp_range": 1 + random.random() * .5,
    }),

    "berkeley": lambda: extend("multires-ridged", {
        "freq": random.randint(12, 16),
        "post_ridges": True,
        "reindex_range": .375 + random.random() * .125,
        "rgb": random.randint(0, 1),
        "sin": 2 * random.random() * 2,
        "with_shadow": 1,
    }),

    "big-data-startup": lambda: extend("dither", "glyphic", {
        "mask": "script",
        "hue_rotation": random.random(),
        "hue_range": .0625 + random.random() * .5,
        "post_saturation": .125 + random.random() * .125,
        "posterize_levels": random.randint(2, 4),
        "saturation": 1.0,
    }),

    "bit-by-bit": lambda: extend("bloom", "crt", "value-mask", {
        "mask": stash("bit-by-bit-mask", random_member([vm.alphanum_binary, vm.alphanum_hex, vm.alphanum_numeric])),
        "freq": [i * stash("bit-by-bit-repeat", random.randint(30, 60))
            for i in masks.mask_shape(stash("bit-by-bit-mask"))[0:2]],
        "with_shadow": random.random(),
    }),

    "bitmask": lambda: extend("bloom", "multires-low", "value-mask", {
        "mask": stash("bitmask-mask", random_member(vm.procedural_members())),
        "freq": [i * stash("bitmask-repeat", random.randint(7, 15))
            for i in masks.mask_shape(stash("bitmask-mask"))[0:2]],
        "ridges": True,
    }),

    "blacklight-fantasy": lambda: extend("bloom", "dither", "invert", "voronoi", {
        "post_hue_rotation": -.125,
        "posterize_levels": 3,
        "rgb": True,
        "voronoi_refract": .5 + random.random() * 1.25,
        "with_sobel": 1,
        "warp_octaves": random.randint(1, 4),
        "warp_range": random.randint(0, 1) * random.random(),
    }),

    "blobby": lambda: extend("funhouse", "invert", "reverb", "outline", {
        "mask": stash("blobby-mask", random_member(vm)),
        "deriv": random.randint(1, 3),
        "distrib": "uniform",
        "freq": [i * stash("blobby-repeat", random.randint(4, 8))
            for i in masks.mask_shape(stash("blobby-mask"))[0:2]],
        "saturation": .25 + random.random() * .5,
        "hue_range": .25 + random.random() * .5,
        "hue_rotation": random.randint(0, 1) * random.random(),
        "spline_order": random.randint(2, 3),
        "warp_freq": random.randint(6, 12),
        "warp_interp": random.randint(1, 3),
        "with_shadow": 1,
    }),

    "blockchain-stock-photo-background": lambda: extend("glitchin-out", "vignette-dark", "value-mask", {
        "freq": random.randint(10, 15) * 15,
        "mask": random_member(["truetype", "binary", "hex", "numeric", "bank_ocr"]),
    }),

    "branemelt": lambda: extend("multires", {
        "freq": random.randint(6, 12),
        "post_reflect_range": .0375 + random.random() * .025,
        "sin": random.randint(32, 64),
    }),

    "branewaves": lambda: extend("bloom", "value-mask", {
        "mask": stash('branewaves-mask', random_member(vm.grid_members())),
        "freq": [int(i * stash("branewaves-repeat", random.randint(5, 10)))
            for i in masks.mask_shape(stash("branewaves-mask"))[0:2]],
        "ridges": True,
        "ripple_freq": 2,
        "ripple_kink": 1.5 + random.random() * 2,
        "ripple_range": .15 + random.random() * .15,
        "spline_order": random.randint(1, 3),
    }),

    "bringing-hexy-back": lambda: extend("bloom", {
        "lattice_drift": 1,
        "point_distrib": "v_hex" if random.randint(0, 1) else "v_hex",
        "point_freq": 10,
        "post_deriv": random.randint(0, 1) * random.randint(1, 3),
        "voronoi_alpha": 0.5,
        "voronoi_refract": random.randint(0, 1) * random.random() * .5,
        "warp_octaves": 1,
        "warp_range": random.random() * .25,
        "with_voronoi": voronoi.range_regions,
    }),

    "broken": lambda: extend("dither", "multires-low", {
        "freq": random.randint(3, 4),
        "lattice_drift": 2,
        "post_brightness": .125,
        "post_saturation": .25,
        "posterize_levels": 3,
        "reindex_range": random.randint(3, 4),
        "rgb": True,
        "speed": .025,
        "with_glowing_edges": 1,
    }),

    "bubble-machine": lambda: extend("maybe-invert", "outline", "wormhole", {
        "corners": True,
        "distrib": "uniform",
        "freq": random.randint(3, 6) * 2,
        "mask": random_member(["h_hex", "v_hex"]),
        "posterize_levels": random.randint(8, 16),
        "reverb_iterations": random.randint(1, 3),
        "spline_order": random.randint(1, 3),
        "with_reverb": random.randint(3, 5),
        "wormhole_stride": .1 + random.random() * .05,
        "wormhole_kink": .5 + random.random() * 4,
    }),

    "bubble-multiverse": lambda: extend("bloom", "random-hue", {
        "point_freq": 10,
        "post_refract_range": .125 + random.random() * .05,
        "voronoi_refract": .625 + random.random() * .25,
        "with_density_map": True,
        "with_shadow": 1.0,
        "with_voronoi": voronoi.flow,
    }),

    "celebrate": lambda: extend("distressed", {
        "brightness_distrib": "ones",
        "hue_range": 1,
        "posterize_levels": random.randint(3, 5),
        "speed": .025,
    }),

    "cell-reflect": lambda: extend("bloom", "dither", "maybe-invert", {
        "point_freq": random.randint(2, 3),
        "post_deriv": random.randint(1, 3),
        "post_reflect_range": random.randint(2, 4) * 5,
        "post_saturation": .5,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_func": random_member(df.all()),
        "voronoi_nth": random.randint(0, 1),
        "with_density_map": True,
        "with_voronoi": voronoi.color_range,
    }),

    "cell-refract": lambda: {
        "point_freq": random.randint(3, 4),
        "post_ridges": True,
        "reindex_range": 1.0 + random.random() * 1.5,
        "rgb": random.randint(0, 1),
        "ridges": True,
        "voronoi_refract": random.randint(8, 12) * .5,
        "with_voronoi": voronoi.range,
    },

    "cell-refract-2": lambda: extend("bloom", "density-map", "voronoi", {
        "point_freq": random.randint(2, 3),
        "post_deriv": random.randint(0, 3),
        "post_refract_range": random.randint(1, 3),
        "post_saturation": .5,
        "voronoi_alpha": .333 + random.random() * .333,
        "with_voronoi": voronoi.color_range,
    }),

    "cell-worms": lambda: extend("bloom", "density-map", "multires-low", "random-hue", "voronoi", "worms", {
        "freq": random.randint(3, 7),
        "hue_range": .125 + random.random() * .875,
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(2, 4),
        "saturation": .125 + random.random() * .25,
        "voronoi_alpha": .75,
        "with_shadow": .75 + random.random() * .25,
        "worms_density": 1500,
        "worms_kink": random.randint(16, 32),
    }),

    "chiral": lambda: extend("sobel", "symmetry", "voronoi", {
        "point_freq": 1,
        "post_reindex_range": .05,
        "post_refract_range": random.randint(18, 36),
        "speed": .025,
        "voronoi_alpha": .95,
        "with_density_map": True,
        "with_voronoi": voronoi.flow,
    }),

    "circulent": lambda: extend("invert", "reverb", "symmetry", "voronoi", "wormhole", {
        "point_distrib": random_member(["spiral"] + [m.name for m in pd.circular_members()]),
        "point_corners": True,
        "voronoi_alpha": .5 + random.random() * .5,
        "wormhole_kink": random.randint(3, 6),
        "wormhole_stride": .05 + random.random() * .05,
    }),

    "classic-desktop": lambda: extend("basic", "lens-warp", {
        "hue_range": .333 + random.random() * .333,
        "lattice_drift": random.random(),
    }),

    "color-flow": lambda: extend("basic-voronoi", {
        "freq": 64,
        "hue_range": 5,
        "with_voronoi": voronoi.color_flow,
    }),

    "conference": lambda: extend("sobel", "value-mask", {
        "freq": 4 * random.randint(6, 12),
        "mask": "halftone",
        "spline_order": 2,
    }),

    "cool-water": lambda: extend("bloom", {
        "distrib": "uniform",
        "freq": 16,
        "hue_range": .05 + random.random() * .05,
        "hue_rotation": .5125 + random.random() * .025,
        "lattice_drift": 1,
        "octaves": 4,
        "reflect_range": .16667 + random.random() * .16667,
        "refract_range": .25 + random.random() * .125,
        "refract_y_from_offset": True,
        "ripple_range": .005 + random.random() * .0025,
        "ripple_kink": random.randint(2, 4),
        "ripple_freq": random.randint(2, 4),
        "warp_range": .0625 + random.random() * .0625,
        "warp_freq": random.randint(2, 3),
    }),

    "corner-case": lambda: extend("basic", "bloom", "dither", "multires-ridged", {
        "corners": True,
        "lattice_drift": random.randint(0, 1),
        "saturation": random.randint(0, 1) * random.random() * .25,
        "spline_order": 0,
        "with_density_map": True,
    }),

    "crooked": lambda: extend("glitchin-out", "starfield", "pixel-sort-angled"),

    "crop-spirals": lambda: {
        "distrib": "laplace",
        "freq": random.randint(4, 6) * 2,
        "hue_range": 1,
        "saturation": .75,
        "mask": random_member(["h_hex", "v_hex"]),
        "reindex_range": .1 + random.random() * .1,
        "spline_order": 2,
        "with_reverb": random.randint(2, 4),
        "with_worms": 3,
        "worms_alpha": .9 + random.random() * .1,
        "worms_density": 500,
        "worms_duration": 1,
        "worms_kink": 2 + random.random(),
        "worms_stride": .333 + random.random() * .333,
        "worms_stride_deviation": .04 + random.random() * .04,
    },

    "crystallize": lambda: extend("bloom", {
        "point_freq": 4,
        "saturation": .25,
        "vignette_brightness": 0,
        "voronoi_alpha": .5,
        "voronoi_func": df.triangular,
        "voronoi_nth": 4,
        "with_voronoi": voronoi.color_range,
        "with_vignette": .5,
    }),

    "cubert": lambda: extend("crt", {
        "freq": random.randint(4, 6),
        "hue_range": .5 + random.random(),
        "point_freq": random.randint(4, 6),
        "point_distrib": "h_hex",
        "voronoi_func": df.triangular,
        "voronoi_inverse": True,
        "with_voronoi": voronoi.color_range,
    }),

    "cubic": lambda: extend("basic", "bloom", "outline", {
        "point_distrib": "concentric",
        "point_freq": random.randint(3, 5),
        "voronoi_alpha": 0.25 + random.random() * .5,
        "voronoi_nth": random.randint(2, 8),
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
    }),

    "cyclic-dilation": lambda: {
        "freq": random.randint(24, 48),
        "hue_range": .25 + random.random() * 1.25,
        "post_reindex_range": random.randint(4, 6),
        "with_voronoi": voronoi.color_range,
    },

    "deadbeef": lambda: extend("bloom", "corrupt", "value-mask", {
        "freq": 6 * random.randint(9, 24),
        "mask": "hex",
    }),

    "deadlock": lambda: extend("outline", {
        "hue_range": random.random(),
        "hue_rotation": random.random(),
        "saturation": random.random(),
        "point_corners": random.randint(0, 1),
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_drift": random.randint(0, 1) * random.random(),
        "point_freq": 4,
        "point_generations": 2,
        "voronoi_func": random.randint(2, 3),
        "voronoi_nth": random.randint(0, 15),
        "voronoi_alpha": .5 + random.random() * .5,
        "sin": random.random() * 2,
        "with_voronoi": voronoi.range,
    }),

    "death-star-plans": lambda: extend("crt", {
        "point_freq": random.randint(3, 4),
        "post_refract_range": 1,
        "posterize_levels": random.randint(3, 5),
        "voronoi_alpha": 1,
        "voronoi_func": random.randint(2, 3),
        "voronoi_nth": random.randint(2, 3),
        "with_voronoi": voronoi.range,
        "with_sobel": random.randint(1, 3),
    }),

    "deep-field": lambda: extend("multires", "funhouse", {
        "distrib": "uniform",
        "freq": random.randint(16, 20),
        "hue_range": 1,
        "mask": "sparser",
        "mask_static": True,
        "lattice_drift": 1,
        "octaves": 5,
        "reduce_max": True,
        "refract_range": 0.25,
        "warp_range": .05,
    }),

    "defocus": lambda: extend("bloom", "multires", {
        "mask": stash('defocus-mask', random_member(vm)),
        "freq": [int(i * stash("defocus-repeat", random.randint(2, 4)))
            for i in masks.mask_shape(stash("defocus-mask"))[0:2]],
        "sin": 10,
    }),

    "density-wave": lambda: extend("basic", {
        "corners": True,
        "reflect_range": random.randint(2, 6),
        "saturation": 0,
        "with_density_map": True,
        "with_shadow": 1,
    }),

    "different": lambda: extend("multires", {
        "freq": random.randint(8, 12),
        "reflect_range": 7.5 + random.random() * 5.0,
        "reindex_range": .25 + random.random() * .25,
        "speed": .025,
        "sin": random.randint(15, 25),
        "warp_range": .0375 * random.random() * .0375,
    }),

    "diffusion-feedback": lambda: extend("aberration", "bloom", "sobel", {
        "corners": True,
        "distrib": "normal",
        "freq": 8,
        "dla_padding": 5,
        "point_distrib": "square",
        "point_freq": 1,
        "saturation": 0,
        "with_conv_feedback": 125,
        "with_density_map": True,
        "with_dla": .75,
        "with_vignette": .75,
    }),

    "distance": lambda: extend("bloom", "multires", {
        "deriv": random.randint(1, 3),
        "distrib": "exp",
        "lattice_drift": 1,
        "saturation": .06125 + random.random() * .125,
        "with_shadow": 1,
    }),

    "dla-cells": lambda: extend("bloom", {
        "dla_padding": random.randint(2, 8),
        "hue_range": random.random() * 1.5,
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(2, 8),
        "voronoi_alpha": random.random(),
        "with_dla": .5 + random.random() * .5,
        "with_voronoi": random_member(voronoi),
    }),

    "dla-forest": lambda: extend("bloom", {
        "dla_padding": random.randint(2, 8),
        "reverb_iterations": random.randint(2, 4),
        "with_dla": 1,
        "with_reverb": random.randint(3, 6),
    }),

    "domain-warp": lambda: extend("multires-ridged", {
        "post_refract_range": .25 + random.random() * .25,
    }),

    "dropout": lambda: extend("maybe-invert", {
        "distrib": "ones",
        "freq": [random.randint(4, 6), random.randint(2, 4)],
        "mask": "dropout",
        "octaves": random.randint(5, 6),
        "post_deriv": 1,
        "reduce_max": True,
        "rgb": True,
        "spline_order": 0,
    }),

    "ears": lambda: {
        "distrib": "uniform",
        "hue_range": random.random() * 2.5,
        "mask": stash('ears-mask', random_member([m for m in vm if m.name != "chess"])),
        "freq": [int(i * stash("ears-repeat", random.randint(3, 6)))
            for i in masks.mask_shape(stash("ears-mask"))[0:2]],
        "with_worms": 3,
        "worms_alpha": .875,
        "worms_density": 188.07,
        "worms_duration": 3.20,
        "worms_stride": 0.40,
        "worms_stride_deviation": 0.31,
        "worms_kink": 6.36,
    },

    "eat-static": lambda: extend("be-kind-rewind", "scanline-error", {
        "distrib": "simplex",
        "freq": 512,
        "saturation": 0,
        "speed": 2.0,
        "wavelet": True,
    }),

    "electric-worms": lambda: extend("bloom", "density-map", "voronoi", "worms", {
        "freq": random.randint(3, 6),
        "lattice_drift": 1,
        "point_freq": 10,
        "voronoi_alpha": .25 + random.random() * .25,
        "voronoi_func": random_member([2, 4, 101]),
        "voronoi_nth": random.randint(0, 3),
        "with_glowing_edges": .75 + random.random() * .25,
        "with_voronoi": voronoi.color_range,
        "with_worms": 5,
        "worms_alpha": .666 + random.random() * .333,
        "worms_density": 1000,
        "worms_duration": 1,
        "worms_kink": random.randint(7, 9),
        "worms_stride_deviation": 16,
    }),

    "emo": lambda: extend("value-mask", "voronoi", {
        "freq": 13 * random.randint(15, 30),
        "mask": "emoji",
        "spline_order": random.randint(0, 2),
        "voronoi_func": random.randint(2, 3),
        "voronoi_refract": .125 + random.random() * .25,
        "with_voronoi": voronoi.range,
    }),

    "emu": lambda: {
        "mask": stash("emu-mask", random_member(enum_range(vm.emoji_00, vm.emoji_26))),
        "distrib": "ones",
        "freq": masks.mask_shape(stash("emu-mask"))[0:2],
        "point_distrib": stash("emu-mask"),
        "spline_order": 0,
        "voronoi_alpha": .5,
        "voronoi_func": random_member(df.all()),
        "voronoi_refract": .125 + random.random() * .125,
        "voronoi_refract_y_from_offset": False,
        "with_voronoi": voronoi.range,
    },

    "entities": lambda: {
        "distrib": "simplex",
        "freq": 6 * random.randint(4, 5) * 2,
        "hue_range": 1.0 + random.random() * 3.0,
        "mask": "invaders_square",
        "refract_range": .075 + random.random() * .075,
        "refract_y_from_offset": True,
        "spline_order": 2,
    },

    "entity": lambda: extend("entities", {
        "corners": True,
        "distrib": "ones",
        "freq": 6,
        "refract_y_from_offset": False,
        "speed": .05,
        "with_sobel": 1,
    }),

    "eyes": lambda: extend("invert", "outline", {
        "corners": True,
        "distrib": random_member(["ones", "uniform"]),
        "hue_range": random.random(),
        "mask": stash('eyes-mask', random_member([m for m in vm if m.name != "chess"])),
        "freq": [int(i * stash("eyes-repeat", random.randint(3, 6)))
            for i in masks.mask_shape(stash("eyes-mask"))[0:2]],
        "ridges": True,
        "spline_order": random.randint(2, 3),
        "warp_freq": 2,
        "warp_octaves": 1,
        "warp_range": random.randint(1, 4) * .5,
        "with_shadow": 1,
    }),

    "fast-eddies": lambda: extend("bloom", "density-map", {
        "hue_range": .25 + random.random() * .75,
        "hue_rotation": random.random(),
        "octaves": random.randint(1, 3),
        "point_freq": random.randint(2, 10),
        "post_contrast": 1.5,
        "post_saturation": .125 + random.random() * .375,
        "ridges": random.randint(0, 1),
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_refract": 1.0,
        "with_shadow": .75 + random.random() * .25,
        "with_voronoi": voronoi.flow,
        "with_worms": 4,
        "worms_alpha": .5 + random.random() * .5,
        "worms_density": 1000,
        "worms_duration": 6,
        "worms_kink": random.randint(125, 375),
    }),

    "figments": lambda: extend("bloom", "funhouse", "multires-low", "wormhole", {
        "freq": 2,
        "hue_range": 2,
        "lattice_drift": 1,
        "voronoi_refract": .5,
        "with_voronoi": voronoi.flow,
        "wormhole_stride": .05,
        "wormhole_kink": 4,
    }),

    "financial-district": lambda: {
        "point_freq": 2,
        "voronoi_func": 2,
        "voronoi_nth": random.randint(1, 3),
        "with_voronoi": voronoi.range_regions,
    },

    "flowbie": lambda: extend("basic", "bloom", "wormhole", {
        "octaves": random.randint(1, 2),
        "with_worms": random.randint(1, 3),
        "refract_range": random.randint(0, 2),
        "wormhole_alpha": .333 + random.random() * .333,
        "wormhole_kink": .25 + random.random() * .25,
        "wormhole_stride": random.random() * 2.5,
        "worms_alpha": .125 + random.random() * .125,
        "worms_stride": .25 + random.random() * .25,
    }),

    "fossil-hunt": lambda: extend("dither", {
        "freq": random.randint(3, 5),
        "lattice_drift": 1.0,
        "refract_range": random.randint(2, 4),
        "refract_y_from_offset": True,
        "point_freq": 10,
        "posterize_levels": random.randint(3, 5),
        "post_saturation": .25,
        "voronoi_alpha": .5,
        "with_outline": 1,
        "with_voronoi": voronoi.color_range,
    }),

    "fractal-forms": lambda: extend("fractal-seed", {
        "worms_kink": random.randint(256, 512),
    }),

    "fractal-seed": lambda: extend("aberration", "basic", "bloom", "density-map", "multires-low", "random-hue", {
        "hue_range": random.random() * random.randint(1, 3),
        "post_saturation": random_member([.05, .25 + random.random() * .25]),
        "ridges": random.randint(0, 1),
        "speed": .05,
        "with_shadow": .5 + random.random() * .5,
        "with_worms": random.randint(4, 5),
        "worms_alpha": .9 + random.random() * .1,
        "worms_density": random.randint(750, 1500),
    }),

    "fractal-smoke": lambda: extend("fractal-seed", {
        "worms_stride": random.randint(128, 256),
    }),

    "fractile": lambda: extend("bloom", "symmetry", {
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": random.randint(2, 10),
        "reverb_iterations": random.randint(2, 4),
        "voronoi_alpha": min(.75 + random.random() * .5, 1),
        "voronoi_func": random_member(df.all()),
        "voronoi_nth": random.randint(0, 3),
        "with_reverb": random.randint(4, 8),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "fundamentals": lambda: extend("density-map", {
        "freq": random.randint(3, 5),
        "point_freq": random.randint(3, 5),
        "post_deriv": random.randint(1, 3),
        "post_saturation": .333 + random.random() * .333,
        "voronoi_func": random.randint(2, 3),
        "voronoi_nth": random.randint(3, 5),
        "voronoi_refract": .0625 + random.random() * .0625,
        "with_voronoi": voronoi.color_range,
    }),

    "funky-glyphs": lambda: {
        "distrib": random_member(["ones", "uniform"]),
        "mask": stash('funky-glyphs-mask', random_member([
            vm.alphanum_binary,
            vm.alphanum_numeric,
            vm.alphanum_hex,
            vm.lcd,
            vm.lcd_binary,
            vm.fat_lcd,
            vm.fat_lcd_binary,
            vm.fat_lcd_numeric,
            vm.fat_lcd_hex
        ])),
        "freq": [int(i * stash("funky-glyphs-repeat", random.randint(1, 6)))
            for i in masks.mask_shape(stash("funky-glyphs-mask"))[0:2]],
        "octaves": random.randint(1, 2),
        "post_refract_range": .125 + random.random() * .125,
        "post_refract_y_from_offset": True,
        "spline_order": random.randint(1, 3),
    },

    "fuzzy-squares": lambda: extend("value-mask", {
        "corners": True,
        "freq": random.randint(6, 24) * 2,
        "post_contrast": 1.5,
        "spline_order": 1,
        "with_worms": 5,
        "worms_alpha": 1,
        "worms_density": 1000,
        "worms_duration": 2.0,
        "worms_stride": .75 + random.random() * .75,
        "worms_stride_deviation": random.random(),
        "worms_kink": 1 + random.random() * 5.0,
    }),

    "fuzzy-swirls": lambda: {
        "freq": random.randint(2, 32),
        "hue_range": random.random() * 2,
        "point_freq": random.randint(8, 10),
        "spline_order": random.randint(1, 3),
        "voronoi_alpha": 0.5 * random.random() * .5,
        "with_voronoi": voronoi.flow,
        "with_worms": random.randint(1, 4),
        "worms_density": 64,
        "worms_duration": 1,
        "worms_kink": 25,
    },

    "fat-led": lambda: extend("bloom", "value-mask", {
        "mask": stash("fat-led-mask", random_member([
            vm.fat_lcd, vm.fat_lcd_binary, vm.fat_lcd_numeric, vm.fat_lcd_hex])),
        "freq": [int(i * stash("fat-led-repeat", random.randint(15, 30)))
            for i in masks.mask_shape(stash("fat-led-mask"))[0:2]],
    }),

    "fuzzy-thorns": lambda: {
        "point_freq": random.randint(2, 4),
        "point_distrib": "waffle",
        "point_generations": 2,
        "voronoi_inverse": True,
        "voronoi_nth": random.randint(6, 12),
        "with_voronoi": voronoi.color_range,
        "with_worms": random.randint(1, 3),
        "worms_density": 500,
        "worms_duration": 1.22,
        "worms_kink": 2.89,
        "worms_stride": 0.64,
        "worms_stride_deviation": 0.11,
    },

    "galalaga": lambda: extend("crt", {
        "composite_zoom": 2,
        "distrib": "uniform",
        "freq": 6 * random.randint(1, 3),
        "glyph_map_zoom": random.randint(15, 25),
        "glyph_map_colorize": True,
        "hue_range": random.random() * 2.5,
        "mask": "invaders_square",
        "spline_order": 0,
        "with_composite": random_member(["invaders_square", "rgb", ""]),
        "with_glyph_map": "invaders_square",
    }),

    "game-show": lambda: extend("be-kind-rewind", {
        "distrib": "normal",
        "freq": random.randint(8, 16) * 2,
        "mask": random_member(["h_tri", "v_tri"]),
        "posterize_levels": random.randint(2, 5),
        "spline_order": 2,
    }),

    "glass-onion": lambda: {
        "point_freq": random.randint(3, 6),
        "post_deriv": random.randint(1, 3),
        "post_refract_range": .5 + random.random() * .25,
        "voronoi_inverse": random.randint(0, 1),
        "with_reverb": random.randint(3, 5),
        "with_voronoi": voronoi.color_range,
    },

    "globules": lambda: extend("multires-low", {
        "distrib": "ones",
        "freq": random.randint(6, 12),
        "hue_range": .25 + random.random() * .5,
        "lattice_drift": 1,
        "mask": "sparse",
        "mask_static": True,
        "reflect_range": 2.5,
        "saturation": .175 + random.random() * .175,
        "speed": .125,
        "with_density_map": True,
        "with_shadow": 1,
    }),

    "glom": lambda: extend("bloom", {
        "freq": 2,
        "hue_range": .25 + random.random() * .25,
        "lattice_drift": 1,
        "octaves": 2,
        "post_reflect_range": random.randint(1, 2) * .5,
        "post_refract_range": random.randint(1, 2) * .5,
        "post_refract_y_from_offset": True,
        "refract_signed_range": False,
        "reflect_range": random.randint(1, 2) * .125,
        "refract_range": random.randint(1, 2) * .125,
        "refract_y_from_offset": True,
        "speed": .025,
        "warp_range": .125 + random.random() * .125,
        "warp_octaves": 1,
        "with_shadow": .75 + random.random() * .25,
    }),

    "glyphic": lambda: extend("maybe-invert", "value-mask", {
        "mask": stash('glyphic-mask', random_member(vm.procedural_members())),
        "freq": masks.mask_shape(stash("glyphic-mask"))[0:2],
        "octaves": random.randint(3, 5),
        "posterize_levels": 1,
        "reduce_max": True,
        "saturation": 0,
        "spline_order": 2,
    }),

    "graph-paper": lambda: extend("bloom", "crt", "sobel", {
        "corners": True,
        "distrib": "ones",
        "freq": random.randint(4, 12) * 2,
        "hue_range": 0,
        "hue_rotation": random.random(),
        "mask": "chess",
        "rgb": True,
        "saturation": 0.27,
        "spline_order": 0,
        "voronoi_alpha": .25 + random.random() * .75,
        "voronoi_refract": random.random() * 2,
        "with_voronoi": voronoi.flow,
    }),

    "grass": lambda: extend("dither", "multires", {
        "freq": random.randint(6, 12),
        "hue_rotation": .25 + random.random() * .05,
        "lattice_drift": 1,
        "saturation": .625 + random.random() * .25,
        "with_worms": 4,
        "worms_alpha": .9,
        "worms_density": 50 + random.random() * 25,
        "worms_duration": 1.125,
        "worms_stride": .875,
        "worms_stride_deviation": .125,
        "worms_kink": .125 + random.random() * .5,
    }),

    "gravy": lambda: extend("bloom", "value-mask", {
        "freq": 24 * random.randint(2, 6),
        "post_deriv": 2,
        "warp_range": .125 + random.random() * .25,
        "warp_octaves": 3,
        "warp_freq": random.randint(2, 4),
        "warp_interp": 3,
    }),

    "groove-is-stored-in-the-heart": lambda: extend("benny-lava", {
        "ripple_range": 1.0 + random.random() * .5,
        "warp_range": 0,
    }),

    "hairy-diamond": lambda: extend("basic", {
        "erosion_worms_alpha": .75 + random.random() * .25,
        "erosion_worms_contraction": .5 + random.random(),
        "erosion_worms_density": random.randint(25, 50),
        "erosion_worms_iterations": random.randint(25, 50),
        "hue_range": random.random(),
        "hue_rotation": random.random(),
        "saturation": random.random(),
        "point_corners": True,
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(3, 6),
        "point_generations": 2,
        "spline_order": random.randint(0, 3),
        "voronoi_func": random_member([2, 3, 101]),
        "voronoi_inverse": True,
        "voronoi_alpha": .25 + random.random() * .5,
        "with_erosion_worms": True,
        "with_voronoi": random_member([m for m in voronoi if m != voronoi.none]),
    }),

    "halt-catch-fire": lambda: extend("glitchin-out", "multires-low", {
        "freq": 2,
        "hue_range": .05,
        "lattice_drift": 1,
        "spline_order": 0,
    }),

    "hex-machine": lambda: extend("multires", {
        "corners": True,
        "distrib": "ones",
        "freq": random.randint(1, 3) * 2,
        "mask": "h_tri",
        "post_deriv": 3,
        "sin": random.randint(-25, 25),
    }),

    "highland": lambda: {
        "deriv": 1,
        "deriv_alpha": 0.25 + random.random() * .125,
        "freq": 12,
        "hue_range": .125 + random.random() * .125,
        "hue_rotation": .925 + random.random() * .05,
        "octaves": 8,
        "point_freq": 8,
        "post_brightness": .125,
        "ridges": True,
        "saturation": .333,
        "voronoi_alpha": .5,
        "voronoi_nth": 3,
        "with_voronoi": voronoi.color_range,
    },

    "hotel-carpet": lambda: extend("basic", "carpet", "dither", "ripples", {
        "ripple_kink": .25 + random.random() * .25,
        "ripple_range": .5 + random.random() * .25,
        "spline_order": 0,
    }),

    "hsv-gradient": lambda: extend("basic", {
        "hue_range": .125 + random.random() * 2.0,
        "lattice_drift": random.random(),
    }),

    "hydraulic-flow": lambda: extend("basic", "bloom", "dither", "maybe-invert", "multires", {
        "deriv": random.randint(0, 1),
        "deriv_alpha": .25 + random.random() * .25,
        "distrib": random_member([m for m in ValueDistribution if m.name not in ("ones", "mids")]),
        "erosion_worms_alpha": .125 + random.random() * .125,
        "erosion_worms_contraction": .75 + random.random() * .5,
        "erosion_worms_density": random.randint(5, 250),
        "erosion_worms_iterations": random.randint(50, 250),
        "hue_range": random.random(),
        "refract_range": random.random(),
        "ridges": random.randint(0, 1),
        "rgb": random.randint(0, 1),
        "saturation": random.random(),
        "with_erosion_worms": True,
        "with_density_map": True,
        "with_shadow": 1,
    }),

    "i-made-an-art": lambda: extend("distressed", "outline", {
        "spline_order": 0,
        "lattice_drift": random.randint(5, 10),
        "hue_range": random.random() * 4,
    }),

    "inkling": lambda: extend("density-map", {
        "distrib": "ones",
        "freq": random.randint(4, 8),
        "mask": "sparse",
        "mask_static": True,
        "point_freq": 4,
        "post_refract_range": .125 + random.random() * .05,
        "post_saturation": 0,
        "post_contrast": 10,
        "ripple_range": .0125 + random.random() * .00625,
        "voronoi_refract": .25 + random.random() * .125,
        "with_fibers": True,
        "with_grime": True,
        "with_scratches": random.randint(0, 1),
        "with_voronoi": voronoi.flow,
    }),

    "interference": lambda: extend("symmetry", {
        "sin": random.randint(250, 500),
        "with_interference": True
    }),

    "is-this-anything": lambda: extend("soup", {
        "point_freq": 1,
        "voronoi_alpha": .875,
        "voronoi_func": random.randint(1, 3),
    }),

    "isoform": lambda: extend("bloom", "maybe-invert", "outline", {
        "hue_range": random.random(),
        "post_deriv": random.randint(0, 1) * random.randint(1, 3),
        "post_refract_range": .125 + random.random() * .125,
        "ridges": random.randint(0, 1),
        "voronoi_alpha": .75 + random.random() * .25,
        "voronoi_func": random_member([2, 3, 101]),
        "voronoi_nth": random.randint(0, 1),
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
    }),

    "jorts": lambda: extend("dither", {
        "freq": [128,512],
        "glyph_map_alpha": .125 + random.random() * .25,
        "glyph_map_zoom": 2,
        "glyph_map_colorize": True,
        "hue_rotation": .5 + random.random() * .05,
        "hue_range": .0625 + random.random() * .0625,
        "post_brightness": -.125,
        "post_contrast": .0625 + random.random() * .0625,
        "post_saturation": .125 + random.random() * .25,
        "warp_freq": 2,
        "warp_range": .0625 + random.random() * .125,
        "warp_octaves": 1,
        "with_glyph_map": vm.v_bar,
    }),

    "jovian-clouds": lambda: {
        "point_freq": random.randint(8, 10),
        "post_saturation": .125 + random.random() * .25,
        "voronoi_alpha": .175 + random.random() * .25,
        "voronoi_refract": 5.0 + random.random() * 3.0,
        "with_shadow": 1.0,
        "with_voronoi": vornoi.flow,
        "with_worms": 4,
        "worms_alpha": .175 + random.random() * .25,
        "worms_density": 500,
        "worms_duration": 2.0,
        "worms_kink": 192,
    },

    "just-refracts-maam": lambda: extend("basic", {
        "corners": True,
        "post_refract_range": random.randint(0, 1) * random.random(),
        "post_ridges": random.randint(0, 1),
        "refract_range": random.randint(2, 4),
        "ridges": random.randint(0, 1),
        "with_shadow": random.randint(0, 1),
    }),

    "knotty-clouds": lambda: extend("bloom", {
        "point_freq": random.randint(6, 10),
        "voronoi_alpha": .125 + random.random() * .25,
        "with_shadow": 1,
        "with_voronoi": voronoi.color_range,
        "with_worms": 1,
        "worms_alpha": .666 + random.random() * .333,
        "worms_density": 1000,
        "worms_duration": 1,
        "worms_kink": 4,
    }),

    "later": lambda: extend("multires", "procedural-mask", {
        "freq": random.randint(8, 16),
        "point_freq": random.randint(4, 8),
        "spline_order": 0,
        "voronoi_refract": random.randint(1, 4) * .5,
        "warp_freq": random.randint(2, 4),
        "warp_interp": 3,
        "warp_octaves": 2,
        "warp_range": .125 + random.random() * .0625,
        "with_glowing_edges": 1,
        "with_voronoi": voronoi.flow,
    }),

    "lattice-noise": lambda: extend("density-map", {
        "deriv": random.randint(1, 3),
        "freq": random.randint(5, 12),
        "octaves": random.randint(1, 3),
        "post_deriv": random.randint(1, 3),
        "ridges": random.randint(0, 1),
        "saturation": random.random(),
        "with_shadow": random.random(),
    }),

    "lcd": lambda: extend("bloom", "invert", "value-mask", {
        "freq": 40 * random.randint(1, 4),
        "mask": random_member(["lcd", "lcd_binary"]),
        "saturation": .05,
        "with_shadow": random.random(),
    }),

    "led": lambda: extend("bloom", "value-mask", {
        "freq": 40 * random.randint(1, 4),
        "mask": random_member(["lcd", "lcd_binary"]),
    }),

    "lightcycle-derby": lambda: extend("bloom", {
        "freq": random.randint(16, 32),
        "rgb": random.randint(0, 1),
        "spline_order": 0,
        "lattice_drift": 1,
        "with_erosion_worms": True,
    }),

    "lost-in-it": lambda: {
        "distrib": "ones",
        "freq": random.randint(36, 48) * 2,
        "mask": random_member(["h_bar", "v_bar"]),
        "ripple_freq": random.randint(3, 4),
        "ripple_range": 1.0 + random.random() * .5,
        "octaves": 3,
    },

    "lowland": lambda: {
        "deriv": 1,
        "deriv_alpha": .5,
        "freq": 3,
        "hue_range": .125 + random.random() * .25,
        "hue_rotation": .875 + random.random() * .125,
        "lattice_drift": 1,
        "octaves": 8,
        "point_freq": 5,
        "post_brightness": .125,
        "saturation": .666,
        "voronoi_alpha": .333,
        "voronoi_inverse": True,
        "voronoi_nth": 0,
        "with_voronoi": voronoi.color_range,
    },

    "lowpoly-regions": lambda: extend("lowpoly", {
        "with_voronoi": voronoi.color_regions,
        "point_freq": random.randint(2, 3),
    }),

    "lowpoly-regions-tri": lambda: extend("lowpoly-regions", {
        "lowpoly_func": df.triangular,
    }),

    "lsd": lambda: extend("density-map", "distressed", "invert", "random-hue", {
        "brightness_distrib": "mids",
        "freq": random.randint(8, 16),
        "hue_range": random.randint(3, 6),
    }),

    "magic-squares": lambda: extend("bloom", "dither", "multires-low", {
        "distrib": random_member([m.value for m in ValueDistribution if m.name not in ("ones", "mids")]),
        "edges": .25 + random.random() * .5,
        "freq": random_member([9, 12, 15, 18]),
        "hue_range": random.random() * .5,
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": random_member([3, 6, 9]),
        "spline_order": 0,
        "voronoi_alpha": .25,
        "with_voronoi": voronoi.color_regions if random.randint(0, 1) else voronoi.none,
    }),

    "magic-smoke": lambda: extend("basic", {
        "octaves": random.randint(1, 3),
        "with_worms": random.randint(1, 2),
        "worms_alpha": 1,
        "worms_density": 750,
        "worms_duration": .25,
        "worms_kink": random.randint(1, 3),
        "worms_stride": random.randint(64, 256),
    }),

    "mcpaint": lambda: {
        "corners": True,
        "distrib": random_member(["ones", "uniform", "normal"]),
        "freq": random.randint(2, 4),
        "glyph_map_colorize": random.randint(0, 1),
        "glyph_map_zoom": random.randint(3, 6),
        "spline_order": 2,
        "with_glyph_map": "mcpaint",
    },

    "metaballs": lambda: {
        "point_drift": 4,
        "point_freq": 10,
        "posterize_levels": 2,
        "with_voronoi": voronoi.flow,
    },

    "midland": lambda: {
        "deriv": 1,
        "deriv_alpha": .25,
        "freq": 6,
        "hue_range": .25 + random.random() * .125,
        "hue_rotation": .875 + random.random() * .1,
        "octaves": 8,
        "point_freq": 5,
        "post_brightness": .125,
        "saturation": .666,
        "voronoi_refract": .125,
        "voronoi_alpha": .5,
        "voronoi_nth": 1,
        "with_voronoi": voronoi.flow,
    },

    "misaligned": lambda: extend("multires", "outline", {
        "distrib": random_member(ValueDistribution),
        "mask": stash('misaligned-mask', random_member(vm)),
        "freq": [int(i * stash("misaligned-repeat", random.randint(2, 4)))
            for i in masks.mask_shape(stash("misaligned-mask"))[0:2]],
        "spline_order": 0,
    }),

    "moire-than-a-feeling": lambda: extend("basic", "wormhole", {
        "octaves": random.randint(1, 2),
        "point_freq": random.randint(1, 3),
        "saturation": 0,
        "with_density_map": True,
        "with_voronoi": voronoi.range if random.randint(0, 1) else voronoi.none,
        "wormhole_kink": 128,
        "wormhole_stride": .0005,
    }),

    "molded-plastic": lambda: extend("color-flow", {
        "point_distrib": pd.random,
        "post_refract_range": random.randint(25, 30),
        "voronoi_func": 1,
        "voronoi_inverse": True,
    }),

    "molten-glass": lambda: extend("bloom", "lens", "woahdude-octave-warp", {
        "post_contrast": 1.5,
        "with_shadow": 1.0,
    }),

    "multires": lambda: {
        "octaves": random.randint(4, 8),
    },

    "multires-low": lambda: {
        "octaves": random.randint(2, 4),
    },

    "multires-ridged": lambda: extend("multires", {
        "ridges": True
    }),

    "multires-voronoi-worms": lambda: {
        "point_freq": random.randint(8, 10),
        "reverb_ridges": False,
        "with_reverb": 2,
        "with_voronoi": random_member([voronoi.none, voronoi.range, voronoi.flow]),
        "with_worms": 1,
        "worms_density": 1000,
    },

    "muppet-skin": lambda: extend("basic", "bloom", {
        "hue_range": random.random() * .5,
        "lattice_drift": random.randint(0, 1) * random.random(),
        "with_worms": 3 if random.randint(0, 1) else 1,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 500,
    }),

    "mycelium": lambda: extend("fractal-seed", "funhouse", random_member(["defocus", "hex-machine"]), {
        "mask_static": True,
        "with_worms": 5,
    }),

    "nausea": lambda: extend("ripples", "value-mask", {
        "mask": stash('nausea-mask', random_member([vm.h_bar, vm.v_bar])),
        "freq": [int(i * stash("nausea-repeat", random.randint(5, 8)))
            for i in masks.mask_shape(stash("nausea-mask"))[0:2]],
        "rgb": True,
        "ripple_kink": 1.25 + random.random() * 1.25,
        "ripple_freq": random.randint(2, 3),
        "ripple_range": 1.25 + random.random(),
        "spline_order": 0,
        "with_aberration": .05 + random.random() * .05,
    }),

    "nerdvana": lambda: extend("bloom", "density-map", "symmetry", {
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(5, 10),
        "reverb_ridges": False,
        "with_voronoi": voronoi.color_range,
        "with_reverb": 2,
        "voronoi_nth": 1,
    }),

    "neon-cambrian": lambda: extend("aberration", "bloom", "wormhole", {
        "hue_range": 1,
        "posterize_levels": 24,
        "with_sobel": 1,
        "with_voronoi": voronoi.flow,
        "wormhole_stride": 0.25,
    }),

    "neon-plasma": lambda: extend("basic", "multires", {
        "lattice_drift": random.randint(0, 1),
        "ridges": True,
    }),

    "noise-blaster": lambda: extend("multires", {
        "freq": random.randint(3, 4),
        "lattice_drift": 1,
        "post_reindex_range": 2,
        "reindex_range": 4,
        "speed": .025,
        "with_shadow": .25 + random.random() * .25,
    }),

    "now": lambda: extend("multires-low", "outline", {
        "freq": random.randint(3, 10),
        "hue_range": random.random(),
        "saturation": .5 + random.random() * .5,
        "lattice_drift": random.randint(0, 1),
        "point_freq": random.randint(3, 10),
        "spline_order": 0,
        "voronoi_refract": random.randint(1, 4) * .5,
        "warp_freq": random.randint(2, 4),
        "warp_interp": 3,
        "warp_octaves": 1,
        "warp_range": .0375 + random.random() * .0375,
        "with_voronoi": voronoi.flow,
    }),

    "numberwang": lambda: extend("bloom", "value-mask", {
        "freq": 6 * random.randint(15, 30),
        "mask": "numeric",
        "warp_range": .25 + random.random() * .75,
        "warp_freq": random.randint(2, 4),
        "warp_interp": 3,
        "warp_octaves": 1,
        "with_false_color": True
    }),

    "octave-rings": lambda: extend("sobel", {
        "corners": True,
        "distrib": "ones",
        "freq": random.randint(1, 3) * 2,
        "mask": "waffle",
        "octaves": random.randint(1, 2),
        "post_reflect_range": random.randint(0, 2) * 5.0,
        "reverb_ridges": False,
        "with_reverb": random.randint(4, 8),
    }),

    "oldschool": lambda: {
        "corners": True,
        "distrib": "ones",
        "freq": random.randint(2, 5) * 2,
        "mask": "chess",
        "rgb": True,
        "speed": .025,
        "spline_order": 0,
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(4, 8),
        "voronoi_refract": random.randint(8, 12) * .5,
        "with_voronoi": voronoi.flow,
    },

    "oracle": lambda: extend("maybe-invert", "snow", "value-mask", {
        "corners": True,
        "freq": [14, 8],
        "mask": "iching",
        "spline_order": 0,
    }),

    "outer-limits": lambda: extend("be-kind-rewind", "dither", "symmetry", {
        "reindex_range": random.randint(8, 16),
        "saturation": 0,
    }),

    "painterly": lambda: {
        "distrib": "uniform",
        "hue_range": .333 + random.random() * .333,
        "mask": stash('painterly-mask', random_member(vm.grid_members())),
        "freq": masks.mask_shape(stash("painterly-mask"))[0:2],
        "octaves": 8,
        "ripple_freq": random.randint(4, 6),
        "ripple_kink": .0625 + random.random() * .125,
        "ripple_range": .0625 + random.random() * .125,
        "spline_order": 1,
        "warp_freq": random.randint(5, 7),
        "warp_octaves": 8,
        "warp_range": .0625 + random.random() * .125,
    },

    "pearlescent": lambda: extend("bloom", {
        "hue_range": random.randint(3, 5),
        "octaves": random.randint(1, 8),
        "point_freq": random.randint(6, 10),
        "post_refract_range": random.randint(0, 1) * (.125 + random.random() * 2.5),
        "ridges": random.randint(0, 1),
        "saturation": .175 + random.random() * .25,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_refract": .75 + random.random() * .5,
        "with_shadow": .333 + random.random() * .333,
        "with_voronoi": voronoi.flow,
    }),

    "plaid": lambda: extend("dither", "multires-low", {
        "deriv": 3,
        "distrib": "ones",
        "hue_range": random.random() * .5,
        "freq": random.randint(3, 6) * 2,
        "mask": "chess",
        "spline_order": random.randint(1, 3),
        "warp_freq": random.randint(2, 3),
        "warp_range": random.random() * .125,
        "warp_octaves": 1,
    }),

    "pluto": lambda: extend("bloom", "dither", "multires-ridged", "voronoi", {
        "freq": random.randint(4, 8),
        "deriv": random.randint(1, 3),
        "deriv_alpha": .333 + random.random() * .333,
        "hue_rotation": .575,
        "point_distrib": pd.random,
        "refract_range": .075 + random.random() * .075,
        "saturation": .125 + random.random() * .075,
        "voronoi_alpha": .75,
        "voronoi_func": 1,
        "voronoi_nth": 2,
        "with_shadow": .5 + random.random() * .25,
        "with_voronoi": voronoi.color_range,
    }),

    "political-map": lambda: extend("bloom", "dither", "outline", {
        "freq": 5,
        "saturation": 0.35,
        "lattice_drift": 1,
        "octaves": 2,
        "posterize_levels": 4,
        "warp_octaves": 8,
        "warp_range": .5,
    }),

    "precision-error": lambda: extend("bloom", "invert", "symmetry", {
        "deriv": random.randint(1, 3),
        "post_deriv": random.randint(1, 3),
        "reflect_range": .75 + random.random() * 2.0,
        "with_density_map": True,
        "with_shadow": 1,
    }),

    "procedural-mask": lambda: extend("bloom", "crt", "value-mask", {
        "mask": stash('procedural-mask-mask', random_member(vm.procedural_members())),
        "freq": [int(i * stash("procedural-mask-repeat", random.randint(10, 20)))
            for i in masks.mask_shape(stash("procedural-mask-mask"))[0:2]],
    }),

    "procedural-muck": lambda: extend("procedural-mask", {
        "freq": random.randint(100, 250),
        "saturation": 0,
        "spline_order": 0,
        "warp_freq": random.randint(2, 5),
        "warp_interp": 2,
        "warp_range": .25 + random.random(),
    }),

    "prophesy": lambda: extend("invert", "value-mask", {
        "freq": 6 * random.randint(3, 4) * 2,
        "mask": "invaders_square",
        "octaves": 2,
        "refract_range": .0625 + random.random() * .0625,
        "refract_y_from_offset": True,
        "saturation": .125 + random.random() * .075,
        "spline_order": 2,
        "posterize_levels": random.randint(4, 8),
        "with_convolve": ["emboss"],
        "with_shadow": .5,
    }),

    "puzzler": lambda: extend("basic", "maybe-invert", {
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": 10,
        "speed": .025,
        "with_voronoi": voronoi.color_regions,
        "with_wormhole": True,
    }),

    "quadrants": lambda: extend("basic", {
        "freq": 2,
        "post_reindex_range": 2,
        "rgb": True,
        "spline_order": random.randint(2, 3),
        "voronoi_alpha": .625,
    }),

    "quilty": lambda: extend("bloom", "dither", {
        "freq": random.randint(2, 6),
        "saturation": random.random() * .5,
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": random.randint(3, 8),
        "spline_order": 0,
        "voronoi_func": random.randint(2, 3),
        "voronoi_nth": random.randint(0, 4),
        "voronoi_refract": random.randint(1, 3) * .5,
        "with_voronoi": random_member([voronoi.regions, voronoi.color_regions]),
    }),

    "random-preset": lambda:
        preset(random_member([m for m in PRESETS if m != "random-preset"])),

    "rasteroids": lambda: extend("bloom", "crt", "sobel", {
        "distrib": random_member(["uniform", "ones"]),
        "freq": 6 * random.randint(2, 3),
        "mask": random_member(vm),
        "spline_order": 0,
        "warp_freq": random.randint(3, 5),
        "warp_octaves": random.randint(3, 5),
        "warp_range": .125 + random.random() * .0625,
    }),

    "redmond": lambda: extend("bloom", "maybe-invert", "snow", "voronoi", {
        "corners": True,
        "distrib": "uniform",
        "freq": 8,
        "hue_range": random.random() * 4.0,
        "mask": "square",
        "point_generations": 2,
        "point_freq": 2,
        "point_distrib": random_member(["chess", "square"]),
        "point_corners": True,
        "reverb_iterations": random.randint(1, 3),
        "spline_order": 0,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_func": random_member([2, 3, 101]),
        "with_reverb": random.randint(3, 6),
    }),

    "refractal": lambda: extend("invert", {
        "lattice_drift": 1,
        "octaves": random.randint(1, 3),
        "point_freq": random.randint(8, 10),
        "post_reflect_range": random.randint(480, 960),
        "sin": random.random() * 10.0,
        "voronoi_alpha": .5 + random.random() * .5,
        "with_voronoi": voronoi.flow,
    }),

    "regional": lambda: extend("glyph-map", "voronoi", {
        "glyph_map_colorize": True,
        "glyph_map_zoom": random.randint(2, 4),
        "hue_range": .25 + random.random(),
        "voronoi_nth": 0,
        "with_voronoi": voronoi.color_regions,
    }),

    "remember-logo": lambda: extend("crt", "density-map", "symmetry", {
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(3, 7),
        "voronoi_alpha": 1.0,
        "voronoi_nth": random.randint(0, 4),
        "post_deriv": 2,
        "with_vignette": .25 + random.random() * .25,
        "with_voronoi": voronoi.regions,
    }),

    "rgb-shadows": lambda: {
        "brightness_distrib": "mids",
        "distrib": "uniform",
        "freq": random.randint(6, 16),
        "hue_range": random.randint(1, 4),
        "lattice_drift": random.random(),
        "saturation_distrib": "ones",
        "with_shadow": 1,
    },

    "ride-the-rainbow": lambda: extend("distressed", "scuff", "swerve-v", {
        "brightness_distrib": "ones",
        "corners": True,
        "distrib": "column_index",
        "freq": random.randint(6, 12),
        "hue_range": .5 + random.random(),
        "saturation_distrib": "ones",
        "spline_order": 0,
    }),

    "ridged-bubbles": lambda: extend("invert", "symmetry", {
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(4, 10),
        "post_ridges": True,
        "reverb_iterations": random.randint(1, 4),
        "rgb": random.randint(0, 1),
        "voronoi_alpha": .333 + random.random() * .333,
        "with_density_map": random.randint(0, 1),
        "with_reverb": random.randint(2, 4),
        "with_voronoi": voronoi.color_regions,
    }),

    "ridged-ridges": lambda: extend("multires-ridged", {
        "freq": random.randint(2, 8),
        "lattice-drift": random.randint(0, 1),
        "post_ridges": True,
        "rgb": random.randint(0, 1),
    }),

    "ripple-effect": lambda: extend("basic", "bloom", "ripples", {
        "lattice_drift": 1,
        "ridges": random.randint(0, 1),
        "sin": 3,
        "with_shadow": .5 + random.random() * .25,
    }),

    "runes-of-arecibo": lambda: extend("value-mask", {
        "mask": stash("runes-mask", random_member([
           vm.arecibo_num, vm.arecibo_bignum, vm.arecibo_nucleotide])),
        "freq": [int(i * stash("runes-repeat", random.randint(20, 40)))
            for i in masks.mask_shape(stash("runes-mask"))[0:2]],
    }),

    "sands-of-time": lambda: {
        "freq": random.randint(3, 5),
        "octaves": random.randint(1, 3),
        "with_worms": random.randint(3, 4),
        "worms_alpha": 1,
        "worms_density": 750,
        "worms_duration": .25,
        "worms-kink": random.randint(1, 2),
        "worms_stride": random.randint(128, 256),
    },

    "satori": lambda: extend("bloom", {
        "freq": random.randint(3, 8),
        "hue_range": random.random(),
        "lattice_drift": 1,
        "point_distrib": random_member([pd.random] + pd.circular_members()),
        "point_freq": random.randint(2, 8),
        "post_ridges": random.randint(0, 1),
        "rgb": random.randint(0, 1),
        "ridges": True,
        "sin": random.random() * 2.5,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_refract": random.randint(6, 12) * .5,
        "with_shadow": 1.0,
        "with_voronoi": voronoi.flow,
    }),

    "sblorp": lambda: extend("invert", {
        "distrib": "ones",
        "freq": random.randint(5, 9),
        "lattice_drift": 1.25 + random.random() * 1.25,
        "mask": "sparse",
        "octaves": random.randint(2, 3),
        "posterize_levels": 1,
        "reduce_max": True,
        "rgb": True,
    }),

    "scribbles": lambda: extend("dither", "sobel", {
        "deriv": random.randint(1, 3),
        "freq": random.randint(4, 8),
        "lattice_drift": random.random(),
        "octaves": 2,
        "post_contrast": 5,
        "post_deriv": random.randint(1, 3),
        "post_saturation": 0,
        "ridges": True,
        "with_density_map": True,
        "with_fibers": True,
        "with_grime": True,
        "with_vignette": .075 + random.random() * .05,
        "with_shadow": random.random(),
    }),

    "seether": lambda: extend("invert", "symmetry", {
        "hue_range": 1.0 + random.random(),
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(4, 6),
        "post_ridges": True,
        "ridges": True,
        "speed": .05,
        "voronoi_alpha": .25 + random.random() * .25,
        "warp_range": .25,
        "warp_octaves": 6,
        "with_glowing_edges": 1,
        "with_reverb": 1,
        "with_shadow": 1,
        "with_voronoi": voronoi.color_regions,
    }),

    "seether-reflect": lambda: extend("seether", {
        "post_reflect_range": random.randint(40, 60),
    }),

    "seether-refract": lambda: extend("seether", {
        "post_refract_range": random.randint(2, 4),
    }),

    "shape-party": lambda: extend("invert", {
        "distrib": "ones",
        "freq": 23,
        "mask": random_member(vm.procedural_members()),
        "point_freq": 2,
        "posterize_levels": 1,
        "rgb": True,
        "spline_order": 2,
        "voronoi_func": 2,
        "voronoi_nth": 1,
        "voronoi_refract": .125 + random.random() * .25,
        "with_aberration": .075 + random.random() * .075,
        "with_bloom": .075 + random.random() * .075,
        "with_voronoi": voronoi.regions,
    }),

    "shatter": lambda: extend("basic", "maybe-invert", "outline", {
        "point_freq": random.randint(3, 5),
        "post_refract_range": 1.0 + random.random(),
        "post_refract_y_from_offset": True,
        "posterize_levels": random.randint(4, 6),
        "rgb": random.randint(0, 1),
        "speed": .05,
        "voronoi_func": random_member([1, 3]),
        "voronoi_inverse": random.randint(0, 1),
        "with_voronoi": voronoi.range_regions,
    }),

    "shimmer": lambda: {
        "deriv": 1,
        "freq": random.randint(4, 8),
        "hue_range": random.randint(2, 4),
        "lattice_drift": 1,
        "point_freq": 10,
        "post_refract_range": random.randint(4, 6),
        "ridges": True,
        "voronoi_alpha": .975 + random.random() * .025,
        "with_voronoi": voronoi.color_flow,
    },

    "shmoo": lambda: extend("invert", "distressed", "outline", {
        "freq": random.randint(4, 6),
        "hue_range": 2 + random.random(),
        "post_saturation": .5 + random.random() * .25,
        "posterize_levels": random.randint(3, 5),
        "rgb": random.randint(0, 1),
        "speed": .025,
    }),

    "sideways": lambda: extend("bloom", "crt", "multires-low", "pixel-sort", {
        "freq": random.randint(6, 12),
        "distrib": "ones",
        "mask": "script",
        "reflect_range": 5.0,
        "saturation": .06125 + random.random() * .125,
        "sin": random.random() * 4,
        "spline_order": random.randint(1, 3),
        "with_shadow": .5 + random.random() * .5,
    }),

    "sine-here-please": lambda: extend("basic", "multires", {
        "sin": 25 + random.random() * 200,
        "with_shadow": 1,
    }),

    "sined-multifractal": lambda: extend("bloom", "multires-ridged", {
        "distrib": "uniform",
        "freq": random.randint(2, 12),
        "hue_range": random.random(),
        "hue_rotation": random.random(),
        "lattice_drift": .75,
        "sin": -3,
    }),

    "skeletal-lace": lambda: extend("wormhole", {
        "lattice_drift": 1,
        "point_freq": 3,
        "voronoi_nth": 1,
        "voronoi_refract": 12.5,
        "with_voronoi": voronoi.flow,
        "wormhole_stride": 0.01,
    }),

    "slimer": lambda: {
        "freq": random.randint(3, 4),
        "hue_range": .5,
        "point_freq": random.randint(1, 3),
        "post_reindex_range": .25 + random.random() * .333,
        "reindex_range": .5 + random.random() * .666,
        "ripple_range": .0125 + random.random() * .016667,
        "speed": .025,
        "voronoi_alpha": .5 + random.random() * .333,
        "voronoi_refract": random.randint(3, 5) * .5,
        "voronoi_refract_y_from_offset": True,
        "warp_range": .0375 + random.random() * .0375,
        "with_voronoi": voronoi.color_regions,
    },

    "smoke-on-the-water": lambda: extend("bloom", "dither", "shadows", {
        "octaves": 8,
        "point_freq": 10,
        "post_saturation": .5,
        "ridges": 1,
        "voronoi_alpha": .5,
        "voronoi_inverse": True,
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
        "with_worms": 5,
        "worms_density": 1000,
    }),

    "soft-cells": lambda: extend("bloom", {
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(4, 8),
        "voronoi_alpha": .5 + random.random() * .5,
        "with_voronoi": voronoi.range_regions,
    }),

    "soften": lambda: extend("bloom", {
        "freq": 2,
        "hue_range": .25 + random.random() * .25,
        "hue_rotation": random.random(),
        "lattice_drift": 1,
        "octaves": random.randint(1, 4),
        "rgb": random.randint(0, 1),
    }),

    "solar": lambda: extend("bloom", "multires", {
        "freq": random.randint(10, 14),
        "hue_range": .225 + random.random() * .05,
        "hue_rotation": .975,
        "reflect_range": .333 + random.random() * .16667,
        "refract_range": .333 + random.random() * .16667,
        "saturation": 4 + random.random() * 2.5,
        "sin": 3,
        "speed": .05,
        "warp_range": .1 + random.random() * .05,
        "warp_freq": 2,
    }),

    "soup": lambda: extend("bloom", "density-map", {
        "point_freq": random.randint(2, 4),
        "post_refract_range": random.randint(4, 6),
        "voronoi_inverse": True,
        "with_shadow": 1.0,
        "with_voronoi": voronoi.flow,
        "with_worms": 5,
        "worms_alpha": .5 + random.random() * .45,
        "worms_density": 500,
        "worms_kink": 4.0 + random.random() * 2.0,
    }),

    "spaghettification": lambda: extend("aberration", "bloom", "density-map", "multires-low", {
        "point_freq": 1,
        "voronoi_inverse": True,
        "warp_range": .5 + random.random() * .5,
        "with_shadow": .75 + random.random() * .25,
        "with_voronoi": voronoi.flow,
        "with_worms": 4,
        "worms_alpha": .75,
        "worms_density": 1500,
        "worms_stride": random.randint(150, 350),
    }),

    "spectrogram": lambda: extend("dither", "filthy", {
        "distrib": "row_index",
        "freq": random.randint(256, 512),
        "hue_range": .5 + random.random() * .5,
        "mask": "bar_code",
        "spline_order": 0,
    }),

    "spiral-clouds": lambda: extend("basic", "multires", "wormhole", {
        "lattice_drift": 1.0,
        "saturation-distrib": "ones",
        "shadow": 1,
        "wormhole_alpha": .333 + random.random() * .333,
        "wormhole_stride": .001 + random.random() * .0005,
        "wormhole_kink": random.randint(40, 50),
    }),

    "spiral-in-spiral": lambda: {
        "point_distrib": pd.spiral if random.randint(0, 1) else pd.rotating,
        "point_freq": 10,
        "reverb_iterations": random.randint(1, 4),
        "with_reverb": random.randint(0, 6),
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
        "with_worms": random.randint(1, 4),
        "worms_density": 500,
        "worms_duration": 1,
        "worms_kink": random.randint(5, 25),
    },

    "spiraltown": lambda: extend("wormhole", {
        "freq": 2,
        "hue_range": 1,
        "reflect_range": random.randint(3, 6) * .5,
        "spline_order": random.randint(1, 3),
        "wormhole_kink": random.randint(5, 20),
        "wormhole_stride": random.random() * .05,
    }),

    "splash": lambda: extend("aberration", "bloom", "tint", "vaseline", {
        "distrib": "ones",
        "freq": 3,
        "lattice_drift": 1,
        "mask": "dropout",
        "octaves": 6,
        "post_deriv": 3,
        "reduce_max": True,
        "rgb": True,
        "spline_order": 3,
    }),

    "splork": lambda: {
        "distrib": "ones",
        "freq": 33,
        "mask": "bank_ocr",
        "point_freq": 2,
        "posterize_levels": 1,
        "rgb": True,
        "spline_order": 2,
        "voronoi_func": 3,
        "voronoi_nth": 1,
        "voronoi_refract": .125,
        "with_voronoi": voronoi.color_range,
    },

    "square-stripes": lambda: {
        "hue_range": random.random(),
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": 2,
        "point_generations": random.randint(2, 3),
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_func": random_member([2, 3, 101]),
        "voronoi_nth": random.randint(1, 3),
        "voronoi_refract": .73,
        "with_voronoi": voronoi.color_range,
    },

    "stackin-bricks": lambda: {
        "point_freq": 10,
        "voronoi_func": df.triangular,
        "voronoi_inverse": True,
        "with_voronoi": voronoi.color_range,
    },

    "star-cloud": lambda: extend("bloom", "sobel", {
        "deriv": 1,
        "freq": 2,
        "hue_range": random.random() * 2.0,
        "point_freq": 10,
        "reflect_range": random.random() + 2.5,
        "spline_order": 2,
        "voronoi_refract": random.randint(2, 4) * .5,
        "with_voronoi": voronoi.flow,
    }),

    "starfield": lambda: extend("aberration", "dither", "bloom", "multires-low", "nebula", {
        "distrib": "exp",
        "freq": random.randint(200, 300),
        "mask": "sparse",
        "mask_static": True,
        "post_brightness": -.333,
        "post_contrast": 3,
        "spline_order": 1,
        "with_vignette": .25 + random.random() * .25,
    }),

    "stepper": lambda: extend("voronoi", "symmetry", "outline", {
        "hue_range": random.random(),
        "saturation": random.random(),
        "point_freq": random.randint(5, 10),
        "point_corners": random.randint(0, 1),
        "point_distrib": random_member(pd.circular_members()),
        "voronoi_func": random_member([2, 3, 101]),
        "voronoi_nth": random.randint(0, 25),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "subpixelator": lambda: extend("basic", "funhouse", "subpixels"),

    "symmetry": lambda: {
        "corners": True,
        "freq": 2,
    },

    "symmetry-lowpoly": lambda: extend("lowpoly", "symmetry", {
        "lowpoly_distrib": random_member(pd.circular_members()),
        "lowpoly_freq": random.randint(4, 15),
    }),

    "teh-matrex-haz-u": lambda: extend("bloom", "crt", {
        "distrib": "exp",
        "freq": (random.randint(2, 4), random.randint(48, 96)),
        "glyph_map_zoom": random.randint(2, 4),
        "hue_rotation": .4 + random.random() * .2,
        "hue_range": .25,
        "lattice_drift": 1,
        "mask": "sparse",
        "post_saturation": 2,
        "spline_order": 1,
        "with_glyph_map": random_member([
            random_member(["binary", "numeric", "hex"]),
            "truetype",
            "ideogram",
            "invaders_square",
            random_member(["fat_lcd", "fat_lcd_binary", "fat_lcd_numeric", "fat_lcd_hex"]),
            "emoji",
        ]),
    }),

    "tensorflower": lambda: extend("bloom", "symmetry", {
        "hue_range": random.random(),
        "point_corners": True,
        "point_distrib": random_member(["square", "h_hex", "v_hex"]),
        "point_freq": 2,
        "rgb": random.randint(0, 1),
        "spline_order": 0,
        "vortex_range": random.randint(8, 25),
        "with_voronoi": voronoi.range_regions,
    }),

    "the-arecibo-response": lambda: extend("snow", "value-mask", {
        "freq": random.randint(42, 210),
        "mask": 'arecibo',
    }),

    "the-data-must-flow": lambda: extend("bloom", {
        "freq": 2,
        "post_contrast": 2,
        "post_deriv": 1,
        "rgb": True,
        "with_worms": 1,
        "worms_alpha": .9 + random.random() * .1,
        "worms_density": 1.5 + random.random(),
        "worms_duration": 1,
        "worms_stride": 8,
        "worms_stride_deviation": 6,
    }),

    "the-inward-spiral": lambda: {
        "freq": random.randint(12, 24),
        "point_freq": 1,
        "voronoi_alpha": 1.0 - (random.randint(0, 1) * random.random() * .125),
        "voronoi_func": random_member(df.all()),
        "with_voronoi": voronoi.color_range,
        "with_worms": random.randint(1, 5),
        "worms_alpha": 1,
        "worms_duration": random.randint(1, 4),
        "worms_density": 500,
        "worms_kink": random.randint(6, 24),
    },

    "time-to-reflect": lambda: extend("symmetry", {
        "post_reflect_range": 5.0,
        "post_ridges": True,
        "reflect_range": random.randint(35, 70),
        "ridges": True,
        "with_shadow": 1.0,
    }),

    "timeworms": lambda: extend("bloom", "density-map", {
        "freq": random.randint(8, 36),
        "hue_range": 0,
        "mask": "sparse",
        "mask_static": True,
        "octaves": random.randint(1, 3),
        "reflect_range": random.randint(0, 1) * random.random() * 2,
        "spline_order": random.randint(1, 3),
        "with_worms": 1,
        "worms_alpha": 1,
        "worms_density": .25,
        "worms_duration": 10,
        "worms_stride": 2,
        "worms_kink": .25 + random.random() * 2.5,
    }),

    "traceroute": lambda: extend("multires", {
        "corners": True,
        "distrib": "ones",
        "freq": random.randint(2, 6),
        "mask": random_member(vm),
        "with_worms": random.randint(1, 3),
        "worms_density": 500,
        "worms_kink": random.randint(5, 25),
    }),

    "tri-hard": lambda: {
        "hue_range": .125 + random.random(),
        "point_freq": random.randint(8, 10),
        "posterize_levels": 6,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_func": random_member([df.octagram, df.triangular, df.hexagram]),
        "voronoi_refract": .333 + random.random() * .333,
        "voronoi_refract_y_from_offset": False,
        "with_outline": 1,
        "with_voronoi": voronoi.color_range,
    },

    "triangular": lambda: extend("multires", "sobel", {
        "corners": True,
        "distrib": random_member(["ones", "uniform"]),
        "freq": random.randint(1, 4) * 2,
        "mask": random_member(["h_tri", "v_tri"]),
    }),

    "tribbles": lambda: extend("bloom", "invert", {
        "freq": random.randint(4, 10),
        "hue_rotation": 0.375 + random.random() * .15,
        "hue_range": 0.125 + random.random() * .125,
        "saturation": .375 + random.random() * .15,
        "octaves": 3,
        "point_distrib": "h_hex",
        "point_freq": random.randint(2, 5) * 2,
        "ridges": True,
        "voronoi_alpha": 0.5 + random.random() * .25,
        "warp_freq": random.randint(2, 4),
        "warp_octaves": random.randint(2, 4),
        "warp_range": 0.025 + random.random() * .005,
        "with_voronoi": voronoi.range_regions,
        "with_worms": 3,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 750,
        "worms_duration": .5,
        "worms_stride_deviation": .5,
    }),

    "triblets": lambda: extend("bloom", "multires", {
        "distrib": "uniform",
        "freq": random.randint(3, 15) * 2,
        "mask": random_member(vm),
        "hue_rotation": 0.875 + random.random() * .15,
        "saturation": .375 + random.random() * .15,
        "warp_octaves": random.randint(1, 2),
        "warp_freq": random.randint(2, 3),
        "warp_range": 0.025 + random.random() * .05,
        "with_worms": 3,
        "worms_alpha": .875 + random.random() * .125,
        "worms_density": 750,
        "worms_duration": .5,
        "worms_stride": .5,
        "worms_stride_deviation": .25,
    }),

    "trominos": lambda: extend("bloom", "crt", "sobel", "value-mask", {
        "freq": 4 * random.randint(25, 50),
        "mask": "tromino",
        "spline_order": 0,
    }),

    "truchet-maze": lambda: extend("value-mask", {
        "freq": 6 * random.randint(50, 100),
        "mask": random_member(["truchet_lines", "truchet_curves"]),
    }),

    "twister": lambda: extend("wormhole", {
        "freq": random.randint(12, 24),
        "octaves": 2,
        "wormhole_kink": 1 + random.random() * 3,
        "wormhole_stride": .0333 + random.random() * .0333,
    }),

    "unicorn-puddle": lambda: extend("bloom", "invert", "multires", "random-hue", {
        "distrib": "uniform",
        "freq": random.randint(8, 12),
        "hue_range": 2.5,
        "lattice_drift": 1,
        "post_contrast": 1.5,
        "reflect_range": .125 + random.random() * .075,
        "ripple_freq": [random.randint(12, 64), random.randint(12, 64)],
        "ripple_kink": .5 + random.random() * .25,
        "ripple_range": .125 + random.random() * .0625,
        "with_light_leak": .5 + random.random() * .25,
        "with_shadow": 1,
    }),

    "unmasked": lambda: {
        "distrib": "uniform",
        "freq": random.randint(15, 30),
        "mask": random_member(vm.procedural_members()),
        "octaves": random.randint(1, 2),
        "post_reindex_range": 1 + random.random() * 1.5,
        "with_sobel": random.randint(0, 1),
    },

    "value-mask": lambda: {
        "distrib": "ones",
        "mask": stash('value-mask-mask', random_member(vm)),
        "freq": [int(i * stash("value-mask-repeat", random.randint(2, 8)))
            for i in masks.mask_shape(stash("value-mask-mask"))[0:2]],
        "spline_order": random.randint(0, 2),
    },

    "vectoroids": lambda: extend("crt", {
        "freq": 25,
        "distrib": "ones",
        "mask": "sparse",
        "mask_static": True,
        "point_freq": 10,
        "point_drift": .25 + random.random() * .75,
        "post_deriv": 1,
        "spline_order": 0,
        "with_voronoi": voronoi.color_regions,
    }),

    "velcro": lambda: extend("wormhole", {
        "freq": 2,
        "hue_range": random.randint(0, 3),
        "octaves": random.randint(1, 2),
        "reflect_range": random.randint(6, 8) * .5,
        "spline_order": random.randint(2, 3),
        "wormhole_stride": random.random() * .0125,
    }),

    "vortex-checkers": lambda: extend("outline", {
        "freq": random.randint(4, 10) * 2,
        "distrib": random_member(["ones", "uniform", "laplace"]),
        "mask": "chess",
        "hue_range": random.random(),
        "saturation": random.random(),
        "posterize": random.randint(10, 15),
        "reverb_iterations": random.randint(2, 4),
        "sin": .5 + random.random(),
        "spline_order": 0,
        "vortex_range": 2.5 + random.random() * 5,
        "with_reverb": random.randint(3, 5),
    }),

    "wall-art": lambda: extend("glyphic", "lowpoly", {
        "angle": random.random() * 360.0,
        "lowpoly_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
    }),

    "warped-cells": lambda: extend("invert", {
        "point_distrib": random_member(pd, vm.nonprocedural_members()),
        "point_freq": random.randint(6, 10),
        "post_ridges": True,
        "voronoi_alpha": .333 + random.random() * .333,
        "warp_range": .25 + random.random() * .25,
        "with_voronoi": voronoi.color_range,
    }),

    "warped-grid": lambda: extend("aberration", "bloom", "sobel", "value-mask", {
        "corners": True,
        "freq": random.randint(4, 48) * 2,
        "hue_range": 3,
        "saturation": 0.27,
        "posterize_levels": 12,
        "spline_order": 0,
        "warp_interp": random.randint(1, 3),
        "warp_freq": random.randint(2, 4),
        "warp_range": .125 + random.random() * .375,
        "warp_octaves": 1,
    }),

    "watercolor": lambda: {
        "post_saturation": .333,
        "warp_range": .5,
        "warp_octaves": 8,
        "with_fibers": True,
        "with_texture": True,
    },

    "whatami": lambda: extend("invert", {
        "freq": random.randint(7, 9),
        "hue_range": 3,
        "post_reindex_range": 2,
        "reindex_range": 2,
        "voronoi_alpha": .75 + random.random() * .125,
        "with_voronoi": voronoi.color_range,
    }),

    "wiggler": lambda: extend("ears", {
        "lattice_drift": 1.0,
        "hue_range": .125 + random.random() * .25,
        "mask": None,
    }),

    "wild-hair": lambda: extend("multires", "erosion-worms", "voronoi", {
        "erosion_worms_density": 25,
        "erosion_worms_alpha": .125 + random.random() * .125,
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(5, 10),
        "saturation": 0,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_nth": 1,
        "with_voronoi": voronoi.range,
        "with_shadow": .75 + random.random() * .25,
    }),

    "wild-kingdom": lambda: extend("bloom", "dither", "maybe-invert", "outline", "random-hue", {
        "freq": 25,
        "lattice_drift": 1,
        "mask": "sparser",
        "mask_static": True,
        "posterize_levels": 3,
        "rgb": True,
        "ridges": True,
        "spline_order": 2,
        "warp_octaves": 2,
        "warp_range": .025,
    }),

    "wireframe": lambda: extend("basic", "bloom", "multires-low", "sobel", {
        "hue_range": random.random(),
        "saturation": random.random(),
        "lattice_drift": random.random(),
        "point_distrib": random_member(pd.grid_members(), vm.nonprocedural_members()),
        "point_freq": random.randint(7, 10),
        "voronoi_alpha": 0.25 + random.random() * .5,
        "voronoi_nth": random.randint(1, 5),
        "warp_octaves": random.randint(1, 3),
        "warp_range": random.randint(0, 1) * random.random() * .25,
        "with_voronoi": voronoi.range_regions,
    }),

    "woahdude-voronoi-refract": lambda: {
        "freq": 4,
        "hue_range": 2,
        "lattice_drift": 1,
        "point_freq": 8,
        "speed": .025,
        "sin": 100,
        "voronoi_alpha": 0.875,
        "voronoi_refract": .333 + random.random() * .333,
        "with_voronoi": voronoi.range,
    },

    "woahdude-octave-warp": lambda: extend("basic", "octave-warp", {
        "hue_range": random.random() * 3.0,
        "sin": random.randint(5, 15),
    }),

    "wooly-bully": lambda: {
        "hue_range": random.random() * 1.5,
        "point_corners": True,
        "point_distrib": random_member(pd.circular_members()),
        "point_freq": random.randint(2, 3),
        "point_generations": 2,
        "reverb_iterations": random.randint(1, 2),
        "refract_range": random.randint(0, 1) * random.random(),
        "voronoi_func": 3,
        "voronoi_nth": random.randint(1, 3),
        "voronoi_alpha": .5 + random.random() * .5,
        "with_reverb": random.randint(0, 2),
        "with_voronoi": voronoi.color_range,
        "with_worms": 4,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 250 + random.random() * 250,
        "worms_duration": 1 + random.random() * 1.5,
        "worms_kink": 5 + random.random() * 2.0,
        "worms_stride": 2.5,
        "worms_stride_deviation": 1.25,
    },

    "wormstep": lambda: extend("basic", "bloom", {
        "corners": True,
        "lattice_drift": random.randint(0, 1),
        "octaves": random.randint(1, 3),
        "with_worms": 4,
        "worms_alpha": .5 + random.random() * .5,
        "worms_density": 500,
        "worms_kink": 1.0 + random.random() * 4.0,
        "worms_stride": 8.0 + random.random() * 4.0,
    }),

}


# Call after setting seed
def bake_presets(seed):
    generators.set_seed(seed)

    global EFFECTS_PRESETS
    EFFECTS_PRESETS = _EFFECTS_PRESETS()

    global PRESETS
    PRESETS = _PRESETS()

    _STASH = {}


def random_member(*collections):
    collection = []

    for c in collections:
        if isinstance(collection, EnumMeta):
            collection += list(c)

        # maybe it's a list of enum members
        elif isinstance(next(iter(c), None), Enum):
            collection += [s[1] for s in sorted([(m.name, m) for m in c])]

        else:
            # make sure order is deterministic
            collection += sorted(c)

    return collection[random.randint(0, len(collection) - 1)]


def enum_range(a, b):
    enum_class = type(a)

    members = []

    for i in range(a.value, b.value + 1):
        members.append(enum_class(i))

    return members


def stash(key, value=None):
    global _STASH
    if value is not None:
        _STASH[key] = value
    return _STASH[key]


def extend(*args):
    args = deque(args)

    settings = {}

    settings['with_convolve'] = set()

    while args:
        arg = args.popleft()

        if isinstance(arg, str):
            these_settings = preset(arg)

        else:
            these_settings = arg

        settings['with_convolve'].update(these_settings.pop('with_convolve', set()))

        settings.update(these_settings)

    del(settings['name'])

    # Convert to a JSON-friendly type
    settings['with_convolve'] = list(settings['with_convolve'])

    return settings


def preset(name):
    """
    Load the named settings.

    The `artmaker` and `artmangler` scripts demonstrate how presets are used.

    :param str name: Name of the preset to load
    :return: dict
    """

    presets = EFFECTS_PRESETS if name in EFFECTS_PRESETS else PRESETS

    settings = presets[name]()

    if "name" not in settings:
        settings["name"] = name

    return settings


bake_presets(None)
