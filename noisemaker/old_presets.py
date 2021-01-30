"""Deprecated library. Please use presets.py.

Presets library for artmaker/artmangler scripts.

Presets may contain any keyword arg accepted by :func:`~noisemaker.effects.post_process()`
"""

from collections import deque

import random

from noisemaker.composer import coin_flip, enum_range, random_member, stash
from noisemaker.constants import (
    DistanceMetric as distance,
    InterpolationType as interp,
    OctaveBlending as blend,
    PointDistribution as point,
    ValueDistribution as distrib,
    ValueMask as mask,
    VoronoiDiagramType as voronoi,
    WormBehavior as worms,
)
from noisemaker.palettes import PALETTES

import noisemaker.masks as masks


# Baked presets go here
EFFECTS_PRESETS = {}

PRESETS = {}

# Use a lambda to permit re-eval with new seed
_EFFECTS_PRESETS = lambda: {  # noqa: E731
    "aberration": lambda: {
        "with_aberration": .025 + random.random() * .0125,
    },

    "be-kind-rewind": lambda: extend("crt", {
        "with_vhs": True,
    }),

    "bloom": lambda: {
        "with_bloom": .075 + random.random() * .075,
    },

    "carpet": lambda: {
        "with_grime": True,
        "with_worms": worms.chaotic,
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
        "warp_interp": interp.constant,
    },

    "crt": lambda: extend("scanline-error", "snow", {
        "with_crt": True,
    }),

    "degauss": lambda: extend("crt", {
        "with_degauss": .0625 + random.random() * .03125,
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
        "deriv": random_member(distance.absolute_members()),
    },

    "falsetto": lambda: {
        "with_false_color": True
    },

    "filthy": lambda: {
        "with_grime": True,
        "with_scratches": coin_flip(),
        "with_stray_hair": True,
    },

    "funhouse": lambda: {
        "warp_freq": [random.randint(2, 3), random.randint(1, 3)],
        "warp_interp": interp.bicubic,
        "warp_octaves": random.randint(1, 4),
        "warp_range": .25 + random.random() * .125,
        "warp_signed_range": False,
    },

    "glitchin-out": lambda: extend("bloom", "corrupt", "crt", {
        "with_glitch": True,
        "with_ticker": (random.random() > .75),
    }),

    "glowing-edges": lambda: {
        "with_glowing_edges": 1.0,
    },

    "glyph-map": lambda: {
        "glyph_map_colorize": coin_flip(),
        "glyph_map_zoom": random.randint(1, 3),
        "with_glyph_map": random_member(set(mask.procedural_members()).intersection(masks.square_masks())),
    },

    "grayscale": lambda: {
        "post_saturation": 0,
    },

    "invert": lambda: {
        "with_convolve": ["invert"],
    },

    "kaleido": lambda: extend("wobble", {
        "kaleido_blend_edges": coin_flip(),
        "kaleido_sdf_sides": random.randint(0, 10),
        "point_freq": 1,
        "with_kaleido": random.randint(5, 32),
    }),

    "lens": lambda: extend("aberration", "vaseline", "tint", {
        "with_vignette": .125 + random.random() * .125,
    }),

    "lens-warp": lambda: {
        "speed": .05,
        "with_lens_warp": .125 + random.random() * .125,
    },

    "light-leak": lambda: extend("bloom", "vignette-bright", {
        "with_light_leak": .333 + random.random() * .333,
    }),

    "lowpoly": lambda: {
        "with_lowpoly": True,
    },

    "mad-multiverse": lambda: extend("kaleido", "voronoi", {
        "with_voronoi": 0,
    }),

    "maybe-invert": lambda: {
        "with_convolve": [] if coin_flip() else ["invert"],
    },

    "maybe-palette": lambda: {
        "with_palette": random_member(PALETTES) if coin_flip() else None,
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
        "speed": .0333,
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
        "with_outline": distance.euclidean,
    },

    "palette": lambda: {
        "with_palette": random_member(PALETTES),
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

    "polar": lambda: extend("kaleido", {
        "with_kaleido": 1,
    }),

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

    "rotate": lambda: {
        "angle": random.random() * 360.0,
    },

    "scanline-error": lambda: {
        "with_scan_error": True,
    },

    "scuff": lambda: {
        "with_scratches": True,
    },

    "shadow": lambda: {
        "with_shadow": .5 + random.random() * .25,
    },

    "shadows": lambda: extend("shadow", "vignette-dark"),

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
        "with_snow": .333 + random.random() * .333,
    },

    "sobel": lambda: extend("maybe-invert", {
        "with_sobel": random_member(distance.all())
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
        "with_composite": random_member(mask.rgb_members()),
    },

    "swerve-h": lambda: {
        "warp_freq": [random.randint(2, 5), 1],
        "warp_interp": interp.bicubic,
        "warp_octaves": 1,
        "warp_range": .5 + random.random() * .5,
    },

    "swerve-v": lambda: {
        "warp_freq": [1, random.randint(2, 5)],
        "warp_interp": interp.bicubic,
        "warp_octaves": 1,
        "warp_range": .5 + random.random() * .5,
    },

    "tensor-tone": lambda: {
        "glyph_map_colorize": coin_flip(),
        "with_glyph_map": mask.halftone,
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
        "point_distrib": point.random if coin_flip() else random_member(point, mask.nonprocedural_members()),
        "point_freq": random.randint(4, 10),
        "voronoi_metric": random_member(distance.all()),
        "voronoi_inverse": coin_flip(),
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

    "wobble": lambda: {
        "with_wobble": .75 + random.random() * .75,
    },

    "wormhole": lambda: {
        "with_wormhole": True,
        "wormhole_stride": .025 + random.random() * .05,
        "wormhole_kink": .5 + random.random(),
    },

    "worms": lambda: {
        "with_worms": random_member(worms.all()),
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": random.randint(250, 500),
        "worms_duration": .5 + random.random(),
        "worms_kink": 1.0 + random.random() * 1.5,
        "worms_stride": random.random() + .5,
        "worms_stride_deviation": random.random() + .5,
    },

}

_PRESETS = lambda: {  # noqa: E731
    "1969": lambda: extend("density-map", "distressed", "posterize-outline", "nerdvana", {
        "point_corners": True,
        "point_distrib": point.circular,
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
        "voronoi_metric": distance.triangular,
        "with_voronoi": voronoi.color_regions,
    }),

    "1985": lambda: extend("spatter", {
        "freq": random.randint(15, 25),
        "reindex_range": .2 + random.random() * .1,
        "rgb": True,
        "spline_order": interp.constant,
        "voronoi_metric": distance.chebyshev,
        "voronoi_refract": .2 + random.random() * .1,
        "with_voronoi": voronoi.range,
    }),

    "2001": lambda: extend("aberration", "bloom", "invert", "value-mask", {
        "freq": 13 * random.randint(10, 20),
        "mask": mask.bank_ocr,
        "posterize_levels": 1,
        "vignette_brightness": 1,
        "with_vignette": 1,
    }),

    "2d-chess": lambda: extend("value-mask", {
        "corners": True,
        "freq": 8,
        "mask": mask.chess,
        "point_corners": True,
        "point_distrib": point.square,
        "point_freq": 8,
        "spline_order": interp.constant,
        "voronoi_alpha": 0.5 + random.random() * .5,
        "voronoi_nth": random.randint(0, 1) * random.randint(0, 63),
        "with_voronoi": voronoi.color_range if coin_flip() \
            else random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),  # noqa E131
    }),

    "abyssal-echoes": lambda: extend("multires-alpha", {
        "distrib": distrib.exp,
        "octaves": 5,
        "post_hue_rotation": random.random(),
        "reflect_range": random.randint(20, 30),
        "rgb": True,
    }),

    "acid": lambda: {
        "freq": random.randint(10, 15),
        "octaves": 8,
        "post_reindex_range": 1.25 + random.random() * 1.25,
        "rgb": True,
    },

    "acid-droplets": lambda: extend("bloom", "density-map", "multires-low", "random-hue", "shadow", {
        "freq": random.randint(10, 15),
        "hue_range": 0,
        "mask": mask.sparse,
        "mask_static": True,
        "post_saturation": .25,
        "reflect_range": 3.75 + random.random() * 3.75,
        "with_palette": None,
    }),

    "acid-grid": lambda: extend("bloom", "funhouse", "sobel", "voronoid", {
        "lattice_drift": coin_flip(),
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_freq": 4,
        "point_generations": 2,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_metric": distance.euclidean,
        "with_voronoi": voronoi.color_range,
    }),

    "acid-wash": lambda: extend("funhouse", "reverb", "symmetry", "shadow", {
        "hue_range": 1,
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(6, 10),
        "post_ridges": True,
        "ridges": True,
        "saturation": .25,
        "voronoi_alpha": .333 + random.random() * .333,
        "warp_octaves": 8,
        "with_voronoi": voronoi.color_range,
    }),

    "activation-signal": lambda: extend("glitchin-out", "value-mask", {
        "freq": 4,
        "mask": mask.white_bear,
        "rgb": coin_flip(),
        "spline_order": interp.constant,
        "with_vhs": coin_flip(),
    }),

    "aesthetic": lambda: extend("be-kind-rewind", "maybe-invert", {
        "corners": True,
        "deriv": random_member([distance.none, distance.euclidean]),
        "distrib": random_member([distrib.column_index, distrib.ones, distrib.row_index]),
        "freq": random.randint(3, 5) * 2,
        "mask": mask.chess,
        "spline_order": interp.constant,
        "with_pre_spatter": True,
    }),

    "alien-terrain-multires": lambda: extend("bloom", "maybe-invert", "multires", "shadow", {
        "deriv": distance.euclidean,
        "deriv_alpha": .333 + random.random() * .333,
        "freq": random.randint(4, 8),
        "lattice_drift": 1,
        "post_saturation": .075 + random.random() * .075,
        "saturation": 2,
    }),

    "alien-terrain-worms": lambda: extend("bloom", "dither", "erosion-worms", "multires-ridged", "shadow", {
        "deriv": distance.euclidean,
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
        "with_voronoi": voronoi.flow,
    }),

    "alien-transmission": lambda: extend("glitchin-out", "sobel", "value-mask", {
        "mask": stash("alien-transmission-mask", random_member(mask.procedural_members())),
        # offset by i * .5 for glitched texture lookup
        "freq": [int(i * .5 + i * stash("alien-transmission-repeat", random.randint(20, 30)))
            for i in masks.mask_shape(stash("alien-transmission-mask"))[0:2]],
    }),

    "analog-glitch": lambda: extend("value-mask", {
        "mask": stash("analog-glitch-mask", random_member([mask.alphanum_hex, mask.lcd, mask.fat_lcd])),
        "deriv": distance.manhattan,
        # offset by i * .5 for glitched texture lookup
        "freq": [i * .5 + i * stash("analog-glitch-repeat", random.randint(20, 30))
            for i in masks.mask_shape(stash("analog-glitch-mask"))[0:2]],
    }),

    "anticounterfeit": lambda: extend("dither", "invert", "wormhole", {
        "freq": 2,
        "point_freq": 1,
        "speed": .025,
        "voronoi_refract": .5,
        "with_fibers": True,
        "with_voronoi": voronoi.flow,
        "with_watermark": True,
        "wormhole_kink": 6,
    }),

    "arcade-carpet": lambda: extend("basic", "dither", {
        "post_hue_rotation": -.125,
        "distrib": distrib.exp,
        "freq": random.randint(75, 125),
        "hue_range": 1,
        "mask": mask.sparser,
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
        "mask": mask.truetype,
        "saturation": random.random() * .125,
    }),

    "aztec-waffles": lambda: extend("maybe-invert", "outline", {
        "freq": 7,
        "point_freq": random.randint(2, 4),
        "point_generations": 2,
        "point_distrib": point.circular,
        "posterize_levels": random.randint(6, 18),
        "reflect_range": random.random(),
        "spline_order": interp.constant,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_nth": random.randint(2, 4),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "bad-map": lambda: extend("splork", {
        "freq": random.randint(30, 50),
        "mask": mask.grid,
        "with_voronoi": voronoi.range,
    }),

    "basic": lambda: extend("maybe-palette", {
        "freq": random.randint(2, 4),
    }),

    "basic-lowpoly": lambda: extend("basic", "lowpoly"),

    "basic-voronoi": lambda: extend("basic", "voronoi"),

    "basic-voronoi-refract": lambda: extend("basic", {
        "hue-range": .25 + random.random() * .5,
        "voronoi_refract": .5 + random.random() * .5,
        "with_voronoi": voronoi.range,
    }),

    "band-together": lambda: extend("shadow", {
        "freq": random.randint(6, 12),
        "reindex_range": random.randint(8, 12),
        "warp_range": .5,
        "warp_octaves": 8,
        "warp_freq": 2,
    }),

    "beneath-the-surface": lambda: extend("bloom", "multires-alpha", "shadow", {
        "freq": 3,
        "hue_range": 2.0 + random.random() * 2.0,
        "octaves": 5,
        "reflect_range": 10.0 + random.random() * 5.0,
        "ridges": True,
    }),

    "benny-lava": lambda: extend("distressed", "maybe-palette", {
        "distrib": distrib.column_index,
        "posterize_levels": 1,
        "warp_range": 1 + random.random() * .5,
    }),

    "berkeley": lambda: extend("multires-ridged", "shadow", {
        "freq": random.randint(12, 16),
        "post_ridges": True,
        "reindex_range": .375 + random.random() * .125,
        "rgb": coin_flip(),
        "sin": 2 * random.random() * 2,
    }),

    "big-data-startup": lambda: extend("dither", "glyphic", {
        "mask": mask.script,
        "hue_rotation": random.random(),
        "hue_range": .0625 + random.random() * .5,
        "post_saturation": .125 + random.random() * .125,
        "posterize_levels": random.randint(2, 4),
        "saturation": 1.0,
    }),

    "bit-by-bit": lambda: extend("bloom", "crt", "value-mask", {
        "mask": stash("bit-by-bit-mask", random_member([mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric])),
        "freq": [i * stash("bit-by-bit-repeat", random.randint(30, 60))
            for i in masks.mask_shape(stash("bit-by-bit-mask"))[0:2]],
    }),

    "bitmask": lambda: extend("bloom", "multires-low", "value-mask", {
        "mask": stash("bitmask-mask", random_member(mask.procedural_members())),
        "freq": [i * stash("bitmask-repeat", random.randint(7, 15))
            for i in masks.mask_shape(stash("bitmask-mask"))[0:2]],
        "ridges": True,
    }),

    "blacklight-fantasy": lambda: extend("bloom", "dither", "invert", "voronoi", {
        "post_hue_rotation": -.125,
        "posterize_levels": 3,
        "rgb": True,
        "voronoi_refract": .5 + random.random() * 1.25,
        "with_sobel": distance.euclidean,
        "warp_octaves": random.randint(1, 4),
        "warp_range": random.randint(0, 1) * random.random(),
    }),

    "blobby": lambda: extend("funhouse", "invert", "reverb", "outline", "shadow", {
        "mask": stash("blobby-mask", random_member(mask)),
        "deriv": random_member(distance.absolute_members()),
        "distrib": distrib.uniform,
        "freq": [i * stash("blobby-repeat", random.randint(4, 8))
            for i in masks.mask_shape(stash("blobby-mask"))[0:2]],
        "saturation": .25 + random.random() * .5,
        "hue_range": .25 + random.random() * .5,
        "hue_rotation": random.randint(0, 1) * random.random(),
        "spline_order": random_member([interp.cosine, interp.bicubic]),
        "warp_freq": random.randint(6, 12),
        "warp_interp": random_member([m for m in interp if m != interp.constant]),
    }),

    "blockchain-stock-photo-background": lambda: extend("glitchin-out", "vignette-dark", "value-mask", {
        "freq": random.randint(10, 15) * 15,
        "mask": random_member(
            [mask.truetype, mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric, mask.bank_ocr]),
    }),

    "branemelt": lambda: extend("multires", {
        "freq": random.randint(6, 12),
        "post_reflect_range": .0375 + random.random() * .025,
        "sin": random.randint(32, 64),
    }),

    "branewaves": lambda: extend("bloom", "value-mask", {
        "mask": stash('branewaves-mask', random_member(mask.grid_members())),
        "freq": [int(i * stash("branewaves-repeat", random.randint(5, 10)))
            for i in masks.mask_shape(stash("branewaves-mask"))[0:2]],
        "ridges": True,
        "ripple_freq": 2,
        "ripple_kink": 1.5 + random.random() * 2,
        "ripple_range": .15 + random.random() * .15,
        "spline_order": random_member([m for m in interp if m != interp.constant]),
    }),

    "bringing-hexy-back": lambda: extend("bloom", {
        "lattice_drift": 1,
        "point_distrib": point.v_hex if coin_flip() else point.h_hex,
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

    "bubble-chamber": lambda: extend("tint", {
        "distrib": distrib.exp,
        "post_contrast": 3 + random.random() * 2.0,
        "with_bloom": .5 + random.random() * .25,
        "with_snow": .125 + random.random() * .0625,
        "with_worms": worms.random,
        "worms_alpha": .75,
        "worms_density": .25 + random.random() * .125,
        "worms_drunken_spin": True,
        "worms_drunkenness": .1 + random.random() * .05,
        "worms_stride_deviation": 5.0 + random.random() * 5.0,
    }),

    "bubble-machine": lambda: extend("maybe-invert", "outline", "wormhole", {
        "corners": True,
        "distrib": distrib.uniform,
        "freq": random.randint(3, 6) * 2,
        "mask": random_member([mask.h_hex, mask.v_hex]),
        "posterize_levels": random.randint(8, 16),
        "reverb_iterations": random.randint(1, 3),
        "spline_order": random_member([m for m in interp if m != interp.constant]),
        "with_reverb": random.randint(3, 5),
        "wormhole_stride": .1 + random.random() * .05,
        "wormhole_kink": .5 + random.random() * 4,
    }),

    "bubble-multiverse": lambda: extend("bloom", "random-hue", "shadow", {
        "point_freq": 10,
        "post_refract_range": .125 + random.random() * .05,
        "speed": .05,
        "voronoi_refract": .625 + random.random() * .25,
        "with_density_map": True,
        "with_voronoi": voronoi.flow,
    }),

    "celebrate": lambda: extend("distressed", "maybe-palette", {
        "brightness_distrib": distrib.ones,
        "hue_range": 1,
        "posterize_levels": random.randint(3, 5),
        "speed": .025,
    }),

    "cell-reflect": lambda: extend("bloom", "dither", "maybe-invert", {
        "point_freq": random.randint(2, 3),
        "post_deriv": random_member(distance.absolute_members()),
        "post_reflect_range": random.randint(2, 4) * 5,
        "post_saturation": .5,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_metric": random_member(distance.all()),
        "voronoi_nth": coin_flip(),
        "with_density_map": True,
        "with_voronoi": voronoi.color_range,
    }),

    "cell-refract": lambda: {
        "point_freq": random.randint(3, 4),
        "post_ridges": True,
        "reindex_range": 1.0 + random.random() * 1.5,
        "rgb": coin_flip(),
        "ridges": True,
        "voronoi_refract": random.randint(8, 12) * .5,
        "with_voronoi": voronoi.range,
    },

    "cell-refract-2": lambda: extend("bloom", "density-map", "voronoi", {
        "point_freq": random.randint(2, 3),
        "post_deriv": random_member([distance.none] + distance.absolute_members()),
        "post_refract_range": random.randint(1, 3),
        "post_saturation": .5,
        "voronoi_alpha": .333 + random.random() * .333,
        "with_voronoi": voronoi.color_range,
    }),

    "cell-worms": lambda: extend("bloom", "density-map", "multires-low", "random-hue", "shadow", "voronoi", "worms", {
        "freq": random.randint(3, 7),
        "hue_range": .125 + random.random() * .875,
        "point_distrib": random_member(point, mask.nonprocedural_members()),
        "point_freq": random.randint(2, 4),
        "saturation": .125 + random.random() * .25,
        "voronoi_alpha": .75,
        "worms_density": 1500,
        "worms_kink": random.randint(16, 32),
    }),

    "circulent": lambda: extend("invert", "reverb", "symmetry", "voronoi", "wormhole", {
        "point_distrib": random_member([point.spiral] + point.circular_members()),
        "point_corners": True,
        "speed": .025,
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

    "concentric": lambda: extend("wobble", {
        "distrib": distrib.ones,
        "freq": 2,
        "mask": mask.h_bar,
        "point_drift": 0,
        "point_freq": random.randint(1, 2),
        "rgb": True,
        "speed": .75,
        "spline_order": interp.constant,
        "voronoi_metric": random_member(distance.absolute_members()),
        "voronoi_refract": random.randint(8, 16),
        "with_voronoi": voronoi.range,
    }),

    "conference": lambda: extend("sobel", "value-mask", {
        "freq": 4 * random.randint(6, 12),
        "mask": mask.halftone,
        "spline_order": interp.cosine,
    }),

    "cool-water": lambda: extend("bloom", {
        "distrib": distrib.uniform,
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
        "lattice_drift": coin_flip(),
        "saturation": random.randint(0, 1) * random.random() * .25,
        "spline_order": interp.constant,
        "with_density_map": True,
    }),

    "cosmic-thread": lambda: extend("bloom", {
        "rgb": True,
        "with_worms": random_member(worms.all()),
        "worms_alpha": .925,
        "worms_density": .125,
        "worms_drunkenness": .125,
        "worms_duration": 125,
    }),

    "crooked": lambda: extend("glitchin-out", "starfield", "pixel-sort-angled"),

    "crop-spirals": lambda: {
        "distrib": distrib.pow_inv_1,
        "freq": random.randint(4, 6) * 2,
        "hue_range": 1,
        "saturation": .75,
        "mask": random_member([mask.h_hex, mask.v_hex]),
        "reindex_range": .1 + random.random() * .1,
        "spline_order": interp.cosine,
        "with_reverb": random.randint(2, 4),
        "with_worms": worms.unruly,
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
        "voronoi_metric": distance.triangular,
        "voronoi_nth": 4,
        "with_voronoi": voronoi.color_range,
        "with_vignette": .5,
    }),

    "cubert": lambda: extend("crt", {
        "freq": random.randint(4, 6),
        "hue_range": .5 + random.random(),
        "point_freq": random.randint(4, 6),
        "point_distrib": point.h_hex,
        "voronoi_metric": distance.triangular,
        "voronoi_inverse": True,
        "with_voronoi": voronoi.color_range,
    }),

    "cubic": lambda: extend("basic", "bloom", "outline", {
        "point_distrib": point.concentric,
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

    "dark-matter": lambda: extend("multires-alpha", {
        "octaves": 5,
        "reflect_range": random.randint(20, 30),
    }),

    "deadbeef": lambda: extend("bloom", "corrupt", "value-mask", {
        "freq": 6 * random.randint(9, 24),
        "mask": mask.alphanum_hex,
    }),

    "deadlock": lambda: extend("outline", {
        "hue_range": random.random(),
        "hue_rotation": random.random(),
        "saturation": random.random(),
        "point_corners": coin_flip(),
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_drift": random.randint(0, 1) * random.random(),
        "point_freq": 4,
        "point_generations": 2,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_nth": random.randint(0, 15),
        "voronoi_alpha": .5 + random.random() * .5,
        "sin": random.random() * 2,
        "with_voronoi": voronoi.range,
    }),

    "death-star-plans": lambda: extend("crt", {
        "point_freq": random.randint(3, 4),
        "post_refract_range": 1,
        "post_refract_y_from_offset": True,
        "posterize_levels": random.randint(3, 5),
        "voronoi_alpha": 1,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_nth": random.randint(2, 3),
        "with_voronoi": voronoi.range,
        "with_sobel": random_member(distance.all()),
    }),

    "deep-field": lambda: extend("multires", "funhouse", {
        "distrib": distrib.uniform,
        "freq": random.randint(16, 20),
        "hue_range": 1,
        "mask": mask.sparser,
        "mask_static": True,
        "lattice_drift": 1,
        "octave_blending": blend.reduce_max,
        "octaves": 5,
        "refract_range": 0.25,
        "warp_freq": [2, 2],
        "warp_range": .025,
        "warp_signed_range": True,
        "with_palette": None,
    }),

    "deeper": lambda: extend("multires-alpha", {
        "hue_range": 1.0,
        "octaves": 8,
    }),

    "defocus": lambda: extend("bloom", "multires", {
        "mask": stash('defocus-mask', random_member(mask)),
        "freq": [int(i * stash("defocus-repeat", random.randint(2, 4)))
            for i in masks.mask_shape(stash("defocus-mask"))[0:2]],
        "sin": 10,
    }),

    "density-wave": lambda: extend("basic", "shadow", {
        "corners": True,
        "reflect_range": random.randint(2, 6),
        "saturation": 0,
        "with_density_map": True,
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
        "freq": 8,
        "dla_padding": 5,
        "point_distrib": point.square,
        "point_freq": 1,
        "saturation": 0,
        "with_conv_feedback": 125,
        "with_density_map": True,
        "with_dla": .75,
        "with_vignette": .75,
    }),

    "distance": lambda: extend("bloom", "multires", "shadow", {
        "deriv": random_member(distance.absolute_members()),
        "distrib": distrib.exp,
        "lattice_drift": 1,
        "saturation": .06125 + random.random() * .125,
    }),

    "dla-cells": lambda: extend("bloom", {
        "dla_padding": random.randint(2, 8),
        "hue_range": random.random() * 1.5,
        "point_distrib": random_member(point, mask.nonprocedural_members()),
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

    "dmt": lambda: {
        "brightness_distrib": distrib.ones,
        "freq": 4,
        "hue_range": 3.5 + random.random() * 2.0,
        "kaleido_sdf_sides": random.randint(0, 10),
        "point_distrib": random_member([point.square, point.waffle]),
        "point_freq": 4,
        "post_refract_range": .075 + random.random() * .075,
        "speed": .025,
        "voronoi_metric": random_member(distance.all()),
        "voronoi_refract": .075 + random.random() * .075,
        "with_kaleido": 4,
        "with_voronoi": voronoi.range,
    },

    "domain-warp": lambda: extend("multires-ridged", {
        "post_refract_range": .25 + random.random() * .25,
    }),

    "dropout": lambda: extend("maybe-invert", {
        "distrib": distrib.ones,
        "freq": [random.randint(4, 6), random.randint(2, 4)],
        "mask": mask.dropout,
        "octave_blending": blend.reduce_max,
        "octaves": random.randint(5, 6),
        "post_deriv": distance.euclidean,
        "rgb": True,
        "spline_order": interp.constant,
    }),

    "ears": lambda: {
        "distrib": distrib.uniform,
        "hue_range": random.random() * 2.5,
        "mask": stash('ears-mask', random_member([m for m in mask if m != mask.chess])),
        "freq": [int(i * stash("ears-repeat", random.randint(3, 6)))
            for i in masks.mask_shape(stash("ears-mask"))[0:2]],
        "with_worms": worms.unruly,
        "worms_alpha": .875,
        "worms_density": 188.07,
        "worms_duration": 3.20,
        "worms_stride": 0.40,
        "worms_stride_deviation": 0.31,
        "worms_kink": 6.36,
    },

    "eat-static": lambda: extend("be-kind-rewind", "scanline-error", {
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
        "voronoi_metric": random_member([distance.manhattan, distance.octagram, distance.triangular]),
        "voronoi_nth": random.randint(0, 3),
        "with_glowing_edges": .75 + random.random() * .25,
        "with_voronoi": voronoi.color_range,
        "with_worms": worms.random,
        "worms_alpha": .666 + random.random() * .333,
        "worms_density": 1000,
        "worms_duration": 1,
        "worms_kink": random.randint(7, 9),
        "worms_stride_deviation": 16,
    }),

    "emo": lambda: extend("value-mask", "voronoi", {
        "freq": 13 * random.randint(15, 30),
        "mask": mask.emoji,
        "spline_order": random_member([m for m in interp if m != m.bicubic]),
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_refract": .125 + random.random() * .25,
        "with_voronoi": voronoi.range,
    }),

    "emu": lambda: {
        "mask": stash("emu-mask", random_member(enum_range(mask.emoji_00, mask.emoji_26))),
        "distrib": distrib.ones,
        "freq": masks.mask_shape(stash("emu-mask"))[0:2],
        "point_distrib": stash("emu-mask"),
        "spline_order": interp.constant,
        "voronoi_alpha": .5,
        "voronoi_metric": random_member(distance.all()),
        "voronoi_refract": .125 + random.random() * .125,
        "voronoi_refract_y_from_offset": False,
        "with_voronoi": voronoi.range,
    },

    "entities": lambda: {
        "freq": 6 * random.randint(4, 5) * 2,
        "hue_range": 1.0 + random.random() * 3.0,
        "mask": mask.invaders_square,
        "refract_range": .075 + random.random() * .075,
        "refract_y_from_offset": True,
        "spline_order": interp.cosine,
    },

    "entity": lambda: extend("entities", {
        "corners": True,
        "distrib": distrib.ones,
        "freq": 6,
        "refract_y_from_offset": False,
        "speed": .05,
        "with_sobel": distance.euclidean,
    }),

    "escape-velocity": lambda: extend("multires-low", {
        "distrib": random_member([distrib.pow_inv_1, distrib.exp, distrib.uniform]),
        "erosion_worms_contraction": .2 + random.random() * .1,
        "erosion_worms_iterations": random.randint(625, 1125),
        "rgb": coin_flip(),
        "with_erosion_worms": True,
    }),

    "explore": lambda: extend("dmt", "kaleido", "lens", {
        "hue_range": .75 + random.random() * .75,
        "brightness_distrib": None,
        "post_refract_range": .75 + random.random() * .75,
        "with_kaleido": random.randint(3, 18),
    }),

    "eyes": lambda: extend("invert", "outline", "shadow", {
        "corners": True,
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "hue_range": random.random(),
        "mask": stash('eyes-mask', random_member([m for m in mask if m != mask.chess])),
        "freq": [int(i * stash("eyes-repeat", random.randint(3, 6)))
            for i in masks.mask_shape(stash("eyes-mask"))[0:2]],
        "ridges": True,
        "spline_order": random_member([interp.cosine, interp.bicubic]),
        "warp_freq": 2,
        "warp_octaves": 1,
        "warp_range": random.randint(1, 4) * .125,
    }),

    "fast-eddies": lambda: extend("bloom", "density-map", "shadow", {
        "hue_range": .25 + random.random() * .75,
        "hue_rotation": random.random(),
        "octaves": random.randint(1, 3),
        "point_freq": random.randint(2, 10),
        "post_contrast": 1.5,
        "post_saturation": .125 + random.random() * .375,
        "ridges": coin_flip(),
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_refract": 1.0,
        "with_voronoi": voronoi.flow,
        "with_worms": worms.chaotic,
        "worms_alpha": .5 + random.random() * .5,
        "worms_density": 1000,
        "worms_duration": 6,
        "worms_kink": random.randint(125, 375),
    }),

    "fat-led": lambda: extend("bloom", "value-mask", {
        "mask": stash("fat-led-mask", random_member([
            mask.fat_lcd, mask.fat_lcd_binary, mask.fat_lcd_numeric, mask.fat_lcd_hex])),
        "freq": [int(i * stash("fat-led-repeat", random.randint(15, 30)))
            for i in masks.mask_shape(stash("fat-led-mask"))[0:2]],
    }),

    "figments": lambda: extend("bloom", "funhouse", "multires-low", "wormhole", {
        "freq": 2,
        "hue_range": 2,
        "lattice_drift": 1,
        "speed": .025,
        "voronoi_refract": .5,
        "with_voronoi": voronoi.flow,
        "wormhole_stride": .05,
        "wormhole_kink": 4,
    }),

    "financial-district": lambda: {
        "point_freq": 2,
        "voronoi_metric": distance.manhattan,
        "voronoi_nth": random.randint(1, 3),
        "with_voronoi": voronoi.range_regions,
    },

    "flowbie": lambda: extend("basic", "bloom", "wormhole", {
        "octaves": random.randint(1, 2),
        "refract_range": random.randint(0, 2),
        "with_worms": random_member([worms.obedient, worms.crosshatch, worms.unruly]),
        "wormhole_alpha": .333 + random.random() * .333,
        "wormhole_kink": .25 + random.random() * .25,
        "wormhole_stride": random.random() * 2.5,
        "worms_alpha": .125 + random.random() * .125,
        "worms_stride": .25 + random.random() * .25,
    }),

    "flux-capacitor": lambda: extend("bloom", "worms", "symmetry", "vignette-dark", {
        "point_freq": 1,
        "post_contrast": 5,
        "speed": .333,
        "voronoi_alpha": 1.0,
        "voronoi_inverse": True,
        "voronoi_metric": distance.triangular,
        "with_voronoi": voronoi.color_range,
        "with_worms": worms.meandering,
        "worms_alpha": .175,
        "worms_density": .25,
        "worms_drunkenness": .125,
        "worms_duration": 10,
        "worms_stride": 1,
        "worms_stride_deviation": 0,
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
        "with_outline": distance.euclidean,
        "with_voronoi": voronoi.color_range,
    }),

    "fractal-forms": lambda: extend("fractal-seed", {
        "worms_kink": random.randint(256, 512),
    }),

    "fractal-seed": lambda: extend("aberration", "basic", "bloom", "density-map", "multires-low", "random-hue", "shadow", {
        "hue_range": random.random() * random.randint(1, 3),
        "post_saturation": random_member([.05, .25 + random.random() * .25]),
        "ridges": coin_flip(),
        "speed": .05,
        "with_palette": None,
        "with_worms": random_member([worms.chaotic, worms.random]),
        "worms_alpha": .9 + random.random() * .1,
        "worms_density": random.randint(750, 1500),
    }),

    "fractal-smoke": lambda: extend("fractal-seed", {
        "worms_stride": random.randint(128, 256),
    }),

    "fractile": lambda: extend("bloom", "symmetry", {
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_freq": random.randint(2, 10),
        "reverb_iterations": random.randint(2, 4),
        "voronoi_alpha": min(.75 + random.random() * .5, 1),
        "voronoi_metric": random_member(distance.all()),
        "voronoi_nth": random.randint(0, 3),
        "with_reverb": random.randint(4, 8),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "fundamentals": lambda: extend("density-map", {
        "freq": random.randint(3, 5),
        "point_freq": random.randint(3, 5),
        "post_deriv": random_member(distance.absolute_members()),
        "post_saturation": .333 + random.random() * .333,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_nth": random.randint(3, 5),
        "voronoi_refract": .0625 + random.random() * .0625,
        "with_voronoi": voronoi.color_range,
    }),

    "funky-glyphs": lambda: {
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "mask": stash('funky-glyphs-mask', random_member([
            mask.alphanum_binary,
            mask.alphanum_numeric,
            mask.alphanum_hex,
            mask.lcd,
            mask.lcd_binary,
            mask.fat_lcd,
            mask.fat_lcd_binary,
            mask.fat_lcd_numeric,
            mask.fat_lcd_hex
        ])),
        "freq": [int(i * stash("funky-glyphs-repeat", random.randint(1, 6)))
            for i in masks.mask_shape(stash("funky-glyphs-mask"))[0:2]],
        "octaves": random.randint(1, 2),
        "post_refract_range": .125 + random.random() * .125,
        "post_refract_y_from_offset": True,
        "spline_order": random_member([m for m in interp if m != interp.constant]),
    },

    "fuzzy-squares": lambda: extend("value-mask", {
        "corners": True,
        "freq": random.randint(6, 24) * 2,
        "post_contrast": 1.5,
        "spline_order": interp.linear,
        "with_worms": worms.random,
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
        "spline_order": random_member([m for m in interp if m != interp.constant]),
        "voronoi_alpha": 0.5 * random.random() * .5,
        "with_voronoi": voronoi.flow,
        "with_worms": random_member([m for m in worms.all() if m != worms.random]),
        "worms_density": 64,
        "worms_duration": 1,
        "worms_kink": 25,
    },

    "fuzzy-thorns": lambda: {
        "point_freq": random.randint(2, 4),
        "point_distrib": point.waffle,
        "point_generations": 2,
        "voronoi_inverse": True,
        "voronoi_nth": random.randint(6, 12),
        "with_voronoi": voronoi.color_range,
        "with_worms": random_member([worms.obedient, worms.crosshatch, worms.unruly]),
        "worms_density": 500,
        "worms_duration": 1.22,
        "worms_kink": 2.89,
        "worms_stride": 0.64,
        "worms_stride_deviation": 0.11,
    },

    "galalaga": lambda: extend("crt", {
        "composite_zoom": 2,
        "distrib": distrib.uniform,
        "freq": 6 * random.randint(1, 3),
        "glyph_map_zoom": random.randint(15, 25),
        "glyph_map_colorize": True,
        "hue_range": random.random() * 2.5,
        "mask": mask.invaders_square,
        "spline_order": interp.constant,
        "with_composite": random_member([mask.invaders_square, mask.rgb, None]),
        "with_glyph_map": mask.invaders_square,
    }),

    "game-show": lambda: extend("be-kind-rewind", "maybe-palette", {
        "freq": random.randint(8, 16) * 2,
        "mask": random_member([mask.h_tri, mask.v_tri]),
        "posterize_levels": random.randint(2, 5),
        "spline_order": interp.cosine,
    }),

    "game-over-man": lambda: extend("galalaga", "glitchin-out", "lens"),

    "glass-darkly": lambda: extend("multires-alpha", {
        "distrib": distrib.pow_inv_1,
        "octaves": 8,
        "post_hue_rotation": .1 + random.random() * .05,
        "post_reflect_range": .95 + random.random() * .1,
        "rgb": True,
    }),

    "glass-onion": lambda: {
        "point_freq": random.randint(3, 6),
        "post_deriv": random_member(distance.absolute_members()),
        "post_refract_range": .5 + random.random() * .25,
        "voronoi_inverse": coin_flip(),
        "with_reverb": random.randint(3, 5),
        "with_voronoi": voronoi.color_range,
    },

    "globules": lambda: extend("multires-low", "shadow", {
        "distrib": distrib.ones,
        "freq": random.randint(6, 12),
        "hue_range": .25 + random.random() * .5,
        "lattice_drift": 1,
        "mask": mask.sparse,
        "mask_static": True,
        "reflect_range": 2.5,
        "saturation": .175 + random.random() * .175,
        "speed": .125,
        "with_density_map": True,
    }),

    "glom": lambda: extend("bloom", "shadow", {
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
    }),

    "glyphic": lambda: extend("maybe-invert", "value-mask", {
        "mask": stash('glyphic-mask', random_member(mask.procedural_members())),
        "freq": masks.mask_shape(stash("glyphic-mask"))[0:2],
        "octave_blending": blend.reduce_max,
        "octaves": random.randint(3, 5),
        "posterize_levels": 1,
        "saturation": 0,
        "spline_order": interp.cosine,
    }),

    "graph-paper": lambda: extend("bloom", "crt", "sobel", {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(4, 12) * 2,
        "hue_range": 0,
        "hue_rotation": random.random(),
        "mask": mask.chess,
        "rgb": True,
        "saturation": 0.27,
        "spline_order": interp.constant,
        "voronoi_alpha": .25 + random.random() * .75,
        "voronoi_refract": random.random() * 2,
        "with_voronoi": voronoi.flow,
    }),

    "grass": lambda: extend("dither", "multires", {
        "freq": random.randint(6, 12),
        "hue_rotation": .25 + random.random() * .05,
        "lattice_drift": 1,
        "saturation": .625 + random.random() * .25,
        "with_worms": worms.chaotic,
        "worms_alpha": .9,
        "worms_density": 50 + random.random() * 25,
        "worms_duration": 1.125,
        "worms_stride": .875,
        "worms_stride_deviation": .125,
        "worms_kink": .125 + random.random() * .5,
    }),

    "gravy": lambda: extend("bloom", "value-mask", {
        "freq": 24 * random.randint(2, 6),
        "post_deriv": distance.manhattan,
        "warp_range": .125 + random.random() * .25,
        "warp_octaves": 3,
        "warp_freq": random.randint(2, 4),
        "warp_interp": interp.bicubic,
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
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(3, 6),
        "point_generations": 2,
        "spline_order": random_member(interp),
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev, distance.triangular]),
        "voronoi_inverse": True,
        "voronoi_alpha": .25 + random.random() * .5,
        "with_erosion_worms": True,
        "with_voronoi": random_member([m for m in voronoi if m != voronoi.none]),
    }),

    "halt-catch-fire": lambda: extend("glitchin-out", "multires-low", {
        "freq": 2,
        "hue_range": .05,
        "lattice_drift": 1,
        "spline_order": interp.constant,
    }),

    "heartburn": lambda: extend("vignette-dark", {
        "freq": random.randint(12, 18),
        "octaves": random.randint(1, 3),
        "point_freq": 1,
        "post_contrast": 10 + random.random() * 5.0,
        "ridges": True,
        "voronoi_alpha": 0.9625,
        "voronoi_inverse": True,
        "voronoi_metric": random_member(distance.all()),
        "with_bloom": 0.25,
        "with_voronoi": 42,
    }),

    "hex-machine": lambda: extend("multires", {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(1, 3) * 2,
        "mask": mask.h_tri,
        "post_deriv": distance.chebyshev,
        "sin": random.randint(-25, 25),
    }),

    "highland": lambda: {
        "deriv": distance.euclidean,
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
        "ripple_kink": .5 + random.random() * .25,
        "ripple_range": .666 + random.random() * .333,
        "spline_order": interp.constant,
    }),

    "hsv-gradient": lambda: extend("basic", {
        "hue_range": .125 + random.random() * 2.0,
        "lattice_drift": random.random(),
    }),

    "hydraulic-flow": lambda: extend("basic", "bloom", "dither", "maybe-invert", "multires", "shadow", {
        "deriv": random_member([distance.none, distance.euclidean]),
        "deriv_alpha": .25 + random.random() * .25,
        "distrib": random_member([m for m in distrib if m not in (distrib.ones, distrib.mids)]),
        "erosion_worms_alpha": .125 + random.random() * .125,
        "erosion_worms_contraction": .75 + random.random() * .5,
        "erosion_worms_density": random.randint(5, 250),
        "erosion_worms_iterations": random.randint(50, 250),
        "hue_range": random.random(),
        "refract_range": random.random(),
        "ridges": coin_flip(),
        "rgb": coin_flip(),
        "saturation": random.random(),
        "with_erosion_worms": True,
        "with_density_map": True,
    }),

    "i-made-an-art": lambda: extend("distressed", "outline", {
        "spline_order": interp.constant,
        "lattice_drift": random.randint(5, 10),
        "hue_range": random.random() * 4,
    }),

    "inkling": lambda: extend("density-map", {
        "distrib": distrib.ones,
        "freq": random.randint(4, 8),
        "mask": mask.sparse,
        "mask_static": True,
        "point_freq": 4,
        "post_refract_range": .125 + random.random() * .05,
        "post_saturation": 0,
        "post_contrast": 10,
        "ripple_range": .0125 + random.random() * .00625,
        "voronoi_refract": .25 + random.random() * .125,
        "with_fibers": True,
        "with_grime": True,
        "with_scratches": coin_flip(),
        "with_voronoi": voronoi.flow,
    }),

    "is-this-anything": lambda: extend("soup", {
        "point_freq": 1,
        "post_refract_range": .125 + random.random() * .0625,
        "voronoi_alpha": .875,
        "voronoi_metric": random_member(distance.absolute_members()),
    }),

    "isoform": lambda: extend("bloom", "maybe-invert", "outline", {
        "hue_range": random.random(),
        "post_deriv": random.randint(0, 1) * random.randint(1, 3),
        "post_refract_range": .125 + random.random() * .125,
        "ridges": coin_flip(),
        "voronoi_alpha": .75 + random.random() * .25,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev, distance.triangular]),
        "voronoi_nth": coin_flip(),
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
    }),

    "its-the-fuzz": lambda: extend("bloom", "multires-low", "muppet-skin", "palette", {
        "lattice_drift": 1.0,
        "with_worms": random_member([worms.obedient, worms.unruly]),
        "worms_alpha": .9 + random.random() * .1,
        "worms_drunkenness": .125 + random.random() * .0625,
    }),

    "jorts": lambda: extend("dither", {
        "freq": [128, 512],
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
        "with_glyph_map": mask.v_bar,
    }),

    "jovian-clouds": lambda: extend("shadow", {
        "point_freq": random.randint(8, 10),
        "post_saturation": .125 + random.random() * .25,
        "voronoi_alpha": .175 + random.random() * .25,
        "voronoi_refract": 5.0 + random.random() * 3.0,
        "with_voronoi": voronoi.flow,
        "with_worms": worms.chaotic,
        "worms_alpha": .175 + random.random() * .25,
        "worms_density": 500,
        "worms_duration": 2.0,
        "worms_kink": 192,
    }),

    "just-refracts-maam": lambda: extend("basic", {
        "corners": True,
        "post_refract_range": random.randint(0, 1) * random.random(),
        "post_ridges": coin_flip(),
        "refract_range": random.randint(2, 4),
        "ridges": coin_flip(),
        "with_shadow": coin_flip() * (.333 + random.random() * .16667),
    }),

    "knotty-clouds": lambda: extend("bloom", "shadow", {
        "point_freq": random.randint(6, 10),
        "voronoi_alpha": .125 + random.random() * .25,
        "with_voronoi": voronoi.color_range,
        "with_worms": worms.obedient,
        "worms_alpha": .666 + random.random() * .333,
        "worms_density": 1000,
        "worms_duration": 1,
        "worms_kink": 4,
    }),

    "later": lambda: extend("multires", "procedural-mask", {
        "freq": random.randint(8, 16),
        "point_freq": random.randint(4, 8),
        "spline_order": interp.constant,
        "voronoi_refract": random.randint(1, 4) * .5,
        "warp_freq": random.randint(2, 4),
        "warp_interp": interp.bicubic,
        "warp_octaves": 2,
        "warp_range": .125 + random.random() * .0625,
        "with_glowing_edges": 1,
        "with_voronoi": voronoi.flow,
    }),

    "lattice-noise": lambda: extend("density-map", "shadow", {
        "deriv": random_member(distance.absolute_members()),
        "freq": random.randint(5, 12),
        "octaves": random.randint(1, 3),
        "post_deriv": random_member(distance.absolute_members()),
        "ridges": coin_flip(),
        "saturation": random.random(),
    }),

    "lcd": lambda: extend("bloom", "invert", "value-mask", "shadow", {
        "freq": 40 * random.randint(1, 4),
        "mask": random_member([mask.lcd, mask.lcd_binary]),
        "saturation": .05,
    }),

    "led": lambda: extend("bloom", "value-mask", {
        "freq": 40 * random.randint(1, 4),
        "mask": random_member([mask.lcd, mask.lcd_binary]),
    }),

    "lightcycle-derby": lambda: extend("bloom", {
        "freq": random.randint(16, 32),
        "rgb": coin_flip(),
        "spline_order": interp.constant,
        "lattice_drift": 1,
        "with_erosion_worms": True,
    }),

    "look-up": lambda: extend("bloom", "multires-alpha", {
        "distrib": distrib.exp,
        "hue_range": .333 + random.random() * .333,
        "lattice_drift": 0,
        "octaves": 8,
        "post_brightness": -.05,
        "post_contrast": 3,
        "post_saturation": .5,
        "ridges": True,
    }),

    "lost-in-it": lambda: {
        "distrib": distrib.ones,
        "freq": random.randint(36, 48) * 2,
        "mask": random_member([mask.h_bar, mask.v_bar]),
        "ripple_freq": random.randint(3, 4),
        "ripple_range": 1.0 + random.random() * .5,
        "octaves": 3,
    },

    "lotus": lambda: extend("dmt", "kaleido", {
        "brightness_distrib": None,
        "hue_range": .75 + random.random() * .75,
        "kaleido_blend_edges": False,
        "kaleido_sdf_sides": 0,
        "post_reflect_range": 10.0 + random.random() * 10.0,
        "with_kaleido": 18,
    }),

    "lowland": lambda: {
        "deriv": distance.euclidean,
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
        "lowpoly_metric": distance.triangular,
    }),

    "lsd": lambda: extend("density-map", "invert", "random-hue", "refract-domain-warp", {
        "brightness_distrib": distrib.ones,
        "freq": random.randint(4, 6),
        "hue_range": random.randint(3, 6),
        "speed": .025,
    }),

    "magic-squares": lambda: extend("bloom", "dither", "multires-low", {
        "distrib": random_member([m.value for m in distance if m not in (distrib.ones, distrib.mids)]),
        "edges": .25 + random.random() * .5,
        "freq": random_member([9, 12, 15, 18]),
        "hue_range": random.random() * .5,
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_freq": random_member([3, 6, 9]),
        "spline_order": interp.constant,
        "voronoi_alpha": .25,
        "with_voronoi": voronoi.color_regions if coin_flip() else voronoi.none,
    }),

    "magic-smoke": lambda: extend("basic", {
        "octaves": random.randint(1, 3),
        "with_worms": random_member([worms.obedient, worms.crosshatch]),
        "worms_alpha": 1,
        "worms_density": 750,
        "worms_duration": .25,
        "worms_kink": random.randint(1, 3),
        "worms_stride": random.randint(64, 256),
    }),

    "mcpaint": lambda: {
        "corners": True,
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "freq": random.randint(2, 4),
        "glyph_map_colorize": coin_flip(),
        "glyph_map_zoom": random.randint(3, 6),
        "spline_order": interp.cosine,
        "with_glyph_map": mask.mcpaint,
    },

    "melting-layers": lambda: {
        "corners": True,
        "distrib": distrib.ones,
        "mask": random_member(mask.procedural_members()),
        "octave_blending": blend.alpha,
        "octaves": 5,
        "refract_range": .1 + random.random() * .05,
        "with_density_map": True,
        "with_outline": distance.euclidean,
    },

    "metaballs": lambda: {
        "point_drift": 4,
        "point_freq": 10,
        "posterize_levels": 2,
        "with_voronoi": voronoi.flow,
    },

    "midland": lambda: {
        "deriv": distance.euclidean,
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
        "distrib": random_member(distrib),
        "mask": stash('misaligned-mask', random_member(mask)),
        "freq": [int(i * stash("misaligned-repeat", random.randint(2, 4)))
            for i in masks.mask_shape(stash("misaligned-mask"))[0:2]],
        "spline_order": interp.constant,
    }),

    "moire-than-a-feeling": lambda: extend("basic", "wormhole", {
        "octaves": random.randint(1, 2),
        "point_freq": random.randint(1, 3),
        "saturation": 0,
        "with_density_map": True,
        "with_voronoi": voronoi.range if coin_flip() else voronoi.none,
        "wormhole_kink": 128,
        "wormhole_stride": .0005,
    }),

    "molded-plastic": lambda: extend("color-flow", {
        "point_distrib": point.random,
        "post_refract_range": random.randint(25, 30),
        "voronoi_metric": distance.euclidean,
        "voronoi_inverse": True,
    }),

    "molten-glass": lambda: extend("bloom", "lens", "woahdude-octave-warp", "shadow", {
        "post_contrast": 1.5,
    }),

    "multires": lambda: extend("basic", {
        "octaves": random.randint(4, 8),
    }),

    "multires-alpha": lambda: extend("multires", {
        "distrib": distrib.exp,
        "lattice_drift": 1,
        "octave_blending": blend.alpha,
        "octaves": 5,
        "with_palette": None,
    }),

    "multires-low": lambda: extend("basic", {
        "octaves": random.randint(2, 4),
    }),

    "multires-ridged": lambda: extend("multires", {
        "ridges": True
    }),

    "multires-voronoi-worms": lambda: {
        "point_freq": random.randint(8, 10),
        "reverb_ridges": False,
        "with_reverb": 2,
        "with_voronoi": random_member([voronoi.none, voronoi.range, voronoi.flow]),
        "with_worms": worms.obedient,
        "worms_density": 1000,
    },

    "muppet-skin": lambda: extend("basic", "bloom", {
        "hue_range": random.random() * .5,
        "lattice_drift": random.randint(0, 1) * random.random(),
        "with_worms": worms.unruly if coin_flip() else worms.obedient,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 500,
        "with_palette": None,
    }),

    "mycelium": lambda: extend("fractal-seed", "funhouse", random_member(["defocus", "hex-machine"]), {
        "mask_static": True,
        "with_worms": worms.random,
    }),

    "nausea": lambda: extend("ripples", "value-mask", {
        "mask": stash('nausea-mask', random_member([mask.h_bar, mask.v_bar])),
        "freq": [int(i * stash("nausea-repeat", random.randint(5, 8)))
            for i in masks.mask_shape(stash("nausea-mask"))[0:2]],
        "rgb": True,
        "ripple_kink": 1.25 + random.random() * 1.25,
        "ripple_freq": random.randint(2, 3),
        "ripple_range": 1.25 + random.random(),
        "spline_order": interp.constant,
        "with_aberration": .05 + random.random() * .05,
    }),

    "nerdvana": lambda: extend("bloom", "density-map", "symmetry", {
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(5, 10),
        "reverb_ridges": False,
        "with_voronoi": voronoi.color_range,
        "with_reverb": 2,
        "voronoi_nth": 1,
    }),

    "neon-cambrian": lambda: extend("aberration", "bloom", "wormhole", {
        "hue_range": 1,
        "posterize_levels": 24,
        "with_sobel": distance.euclidean,
        "with_voronoi": voronoi.flow,
        "wormhole_stride": 0.25,
    }),

    "neon-plasma": lambda: extend("multires", {
        "lattice_drift": coin_flip(),
        "ridges": True,
    }),

    "noise-blaster": lambda: extend("multires", "shadow", {
        "freq": random.randint(3, 4),
        "lattice_drift": 1,
        "post_reindex_range": 2,
        "reindex_range": 4,
        "speed": .025,
    }),

    "noodler": lambda: extend("multires-alpha", "maybe-palette", {
        "distrib": distrib.uniform,
        "lattice_drift": 0,
        "mask": mask.sparse,
        "octaves": 5,
        "reindex_range": 3.0 + random.random() * 2.5,
    }),

    "now": lambda: extend("multires-low", "outline", {
        "freq": random.randint(3, 10),
        "hue_range": random.random(),
        "saturation": .5 + random.random() * .5,
        "lattice_drift": coin_flip(),
        "point_freq": random.randint(3, 10),
        "spline_order": interp.constant,
        "voronoi_refract": random.randint(1, 4) * .5,
        "warp_freq": random.randint(2, 4),
        "warp_interp": interp.bicubic,
        "warp_octaves": 1,
        "warp_range": .0375 + random.random() * .0375,
        "with_voronoi": voronoi.flow,
    }),

    "numberwang": lambda: extend("bloom", "value-mask", {
        "freq": 6 * random.randint(15, 30),
        "mask": mask.alphanum_numeric,
        "warp_range": .25 + random.random() * .75,
        "warp_freq": random.randint(2, 4),
        "warp_interp": interp.bicubic,
        "warp_octaves": 1,
        "with_false_color": True
    }),

    "octave-blend": lambda: extend("multires-alpha", {
        "corners": True,
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "freq": random.randint(2, 5),
        "lattice_drift": 0,
        "mask": random_member(mask.procedural_members()),
        "spline_order": interp.constant,
    }),

    "octave-rings": lambda: extend("sobel", {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(1, 3) * 2,
        "mask": mask.waffle,
        "octaves": random.randint(1, 2),
        "post_reflect_range": random.randint(0, 2) * 5.0,
        "reverb_ridges": False,
        "with_reverb": random.randint(4, 8),
    }),

    "oldschool": lambda: {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(2, 5) * 2,
        "mask": mask.chess,
        "rgb": True,
        "speed": .05,
        "spline_order": interp.constant,
        "point_distrib": random_member(point, mask.nonprocedural_members()),
        "point_freq": random.randint(4, 8),
        "voronoi_refract": random.randint(8, 12) * .5,
        "with_voronoi": voronoi.flow,
    },

    "oracle": lambda: extend("maybe-invert", "snow", "value-mask", {
        "corners": True,
        "freq": [14, 8],
        "mask": mask.iching,
        "spline_order": interp.constant,
    }),

    "outer-limits": lambda: extend("be-kind-rewind", "dither", "symmetry", {
        "reindex_range": random.randint(8, 16),
        "saturation": 0,
    }),

    "oxidize": lambda: extend("bloom", "shadow", {
        "freq": 2,
        "hue_range": .875 + random.random() * .25,
        "lattice_drift": 1,
        "octave_blending": blend.reduce_max,
        "octaves": 8,
        "post_brightness": -.125,
        "post_contrast": 1.25,
        "post_refract_range": .125 + random.random() * .06125,
        "post_saturation": .5,
        "speed": .05,
    }),

    "painterly": lambda: {
        "distrib": distrib.uniform,
        "hue_range": .333 + random.random() * .333,
        "mask": stash('painterly-mask', random_member(mask.grid_members())),
        "freq": masks.mask_shape(stash("painterly-mask"))[0:2],
        "octaves": 8,
        "ripple_freq": random.randint(4, 6),
        "ripple_kink": .0625 + random.random() * .125,
        "ripple_range": .0625 + random.random() * .125,
        "spline_order": interp.linear,
        "warp_freq": random.randint(5, 7),
        "warp_octaves": 8,
        "warp_range": .0625 + random.random() * .125,
    },

    "pearlescent": lambda: extend("bloom", "shadow", {
        "hue_range": random.randint(3, 5),
        "octaves": random.randint(1, 8),
        "point_freq": random.randint(6, 10),
        "post_refract_range": random.randint(0, 1) * (.125 + random.random() * 1.25),
        "ridges": coin_flip(),
        "saturation": .175 + random.random() * .25,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_refract": .75 + random.random() * .5,
        "with_voronoi": voronoi.flow,
    }),

    "plaid": lambda: extend("dither", "multires-low", {
        "deriv": distance.chebyshev,
        "distrib": distrib.ones,
        "hue_range": random.random() * .5,
        "freq": random.randint(3, 6) * 2,
        "mask": mask.chess,
        "spline_order": random.randint(1, 3),
        "warp_freq": random.randint(2, 3),
        "warp_range": random.random() * .125,
        "warp_octaves": 1,
    }),

    "pluto": lambda: extend("bloom", "dither", "multires-ridged", "shadow", "voronoi", {
        "freq": random.randint(4, 8),
        "deriv": random_member(distance.absolute_members()),
        "deriv_alpha": .333 + random.random() * .333,
        "hue_rotation": .575,
        "point_distrib": point.random,
        "refract_range": .075 + random.random() * .075,
        "saturation": .125 + random.random() * .075,
        "voronoi_alpha": .75,
        "voronoi_metric": distance.euclidean,
        "voronoi_nth": 2,
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

    "precision-error": lambda: extend("bloom", "invert", "shadow", "symmetry", {
        "deriv": random_member(distance.absolute_members()),
        "post_deriv": random_member(distance.absolute_members()),
        "reflect_range": .75 + random.random() * 2.0,
        "with_density_map": True,
    }),

    "procedural-mask": lambda: extend("bloom", "crt", "value-mask", {
        "mask": stash('procedural-mask-mask', random_member(mask.procedural_members())),
        "freq": [int(i * stash("procedural-mask-repeat", random.randint(10, 20)))
            for i in masks.mask_shape(stash("procedural-mask-mask"))[0:2]],
    }),

    "procedural-muck": lambda: extend("procedural-mask", {
        "freq": random.randint(100, 250),
        "saturation": 0,
        "spline_order": interp.constant,
        "warp_freq": random.randint(2, 5),
        "warp_interp": interp.cosine,
        "warp_range": .25 + random.random(),
    }),

    "prophesy": lambda: extend("invert", "shadow", "value-mask", {
        "freq": 6 * random.randint(3, 4) * 2,
        "mask": mask.invaders_square,
        "octaves": 2,
        "refract_range": .0625 + random.random() * .0625,
        "refract_y_from_offset": True,
        "saturation": .125 + random.random() * .075,
        "spline_order": interp.cosine,
        "posterize_levels": random.randint(4, 8),
        "with_convolve": ["emboss"],
        "with_palette": None,
    }),

    "puzzler": lambda: extend("basic", "maybe-invert", {
        "point_distrib": random_member(point, mask.nonprocedural_members()),
        "point_freq": 10,
        "speed": .025,
        "with_voronoi": voronoi.color_regions,
        "with_wormhole": True,
    }),

    "quadrants": lambda: extend("basic", {
        "freq": 2,
        "post_reindex_range": 2,
        "rgb": True,
        "spline_order": random_member([interp.cosine, interp.bicubic]),
        "voronoi_alpha": .625,
    }),

    "quilty": lambda: extend("bloom", "dither", {
        "freq": random.randint(2, 6),
        "saturation": random.random() * .5,
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_freq": random.randint(3, 8),
        "spline_order": interp.constant,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
        "voronoi_nth": random.randint(0, 4),
        "voronoi_refract": random.randint(1, 3) * .5,
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
    }),

    "random-preset": lambda:
        preset(random_member([m for m in PRESETS if m != "random-preset"])),

    "rasteroids": lambda: extend("bloom", "crt", "sobel", {
        "distrib": random_member([distrib.uniform, distrib.ones]),
        "freq": 6 * random.randint(2, 3),
        "mask": random_member(mask),
        "spline_order": interp.constant,
        "warp_freq": random.randint(3, 5),
        "warp_octaves": random.randint(3, 5),
        "warp_range": .125 + random.random() * .0625,
    }),

    "redmond": lambda: extend("bloom", "maybe-invert", "snow", "voronoi", {
        "corners": True,
        "distrib": distrib.uniform,
        "freq": 8,
        "hue_range": random.random() * 4.0,
        "mask": mask.square,
        "point_generations": 2,
        "point_freq": 2,
        "point_distrib": random_member([point.chess, point.square]),
        "point_corners": True,
        "reverb_iterations": random.randint(1, 3),
        "spline_order": interp.constant,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev, distance.triangular]),
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
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(3, 7),
        "voronoi_alpha": 1.0,
        "voronoi_nth": random.randint(0, 4),
        "post_deriv": distance.manhattan,
        "with_vignette": .25 + random.random() * .25,
        "with_voronoi": voronoi.regions,
    }),

    "rgb-shadows": lambda: {
        "brightness_distrib": distrib.mids,
        "distrib": distrib.uniform,
        "freq": random.randint(6, 16),
        "hue_range": random.randint(1, 4),
        "lattice_drift": random.random(),
        "saturation_distrib": distrib.ones,
        "with_shadow": 1,
    },

    "ride-the-rainbow": lambda: extend("distressed", "scuff", "swerve-v", {
        "brightness_distrib": distrib.ones,
        "corners": True,
        "distrib": distrib.column_index,
        "freq": random.randint(6, 12),
        "hue_range": .9,
        "saturation_distrib": distrib.ones,
        "spline_order": interp.constant,
    }),

    "ridged-bubbles": lambda: extend("invert", "symmetry", {
        "point_distrib": random_member(point, mask.nonprocedural_members()),
        "point_freq": random.randint(4, 10),
        "post_ridges": True,
        "reverb_iterations": random.randint(1, 4),
        "rgb": coin_flip(),
        "voronoi_alpha": .333 + random.random() * .333,
        "with_density_map": coin_flip(),
        "with_reverb": random.randint(2, 4),
        "with_voronoi": voronoi.color_range,
    }),

    "ridged-ridges": lambda: extend("multires-ridged", {
        "freq": random.randint(2, 8),
        "lattice-drift": coin_flip(),
        "post_ridges": True,
        "rgb": coin_flip(),
    }),

    "ripple-effect": lambda: extend("basic", "bloom", "ripples", "shadow", {
        "lattice_drift": 1,
        "ridges": coin_flip(),
        "sin": 3,
    }),

    "runes-of-arecibo": lambda: extend("value-mask", {
        "mask": stash("runes-mask", random_member([
           mask.arecibo_num, mask.arecibo_bignum, mask.arecibo_nucleotide])),
        "freq": [int(i * stash("runes-repeat", random.randint(20, 40)))
            for i in masks.mask_shape(stash("runes-mask"))[0:2]],
    }),

    "sands-of-time": lambda: {
        "freq": random.randint(3, 5),
        "octaves": random.randint(1, 3),
        "with_worms": random_member([worms.unruly, worms.chaotic]),
        "worms_alpha": 1,
        "worms_density": 750,
        "worms_duration": .25,
        "worms_kink": random.randint(1, 2),
        "worms_stride": random.randint(128, 256),
    },

    "satori": lambda: extend("bloom", "maybe-palette", "shadow", "wobble", {
        "freq": random.randint(3, 8),
        "hue_range": random.random(),
        "lattice_drift": 1,
        "point_distrib": random_member([point.random] + point.circular_members()),
        "point_freq": random.randint(2, 8),
        "post_ridges": coin_flip(),
        "rgb": coin_flip(),
        "ridges": True,
        "sin": random.random() * 2.5,
        "speed": .05,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_refract": random.randint(6, 12) * .5,
        "with_voronoi": voronoi.flow,
    }),

    "sblorp": lambda: extend("invert", {
        "distrib": distrib.ones,
        "freq": random.randint(5, 9),
        "lattice_drift": 1.25 + random.random() * 1.25,
        "mask": mask.sparse,
        "octave_blending": blend.reduce_max,
        "octaves": random.randint(2, 3),
        "posterize_levels": 1,
        "rgb": True,
    }),

    "sbup": lambda: extend("distressed", "falsetto", "palette", {
        "distrib": distrib.ones,
        "freq": 3,
        "mask": mask.square,
        "posterize_levels": random.randint(1, 2),
        "warp_range": 1.5 + random.random(),
    }),

    "scribbles": lambda: extend("dither", "shadow", "sobel", {
        "deriv": random_member(distance.absolute_members()),
        "freq": random.randint(4, 8),
        "lattice_drift": random.random(),
        "octaves": 2,
        "post_contrast": 5,
        "post_deriv": random_member(distance.absolute_members()),
        "post_saturation": 0,
        "ridges": True,
        "with_density_map": True,
        "with_fibers": True,
        "with_grime": True,
        "with_vignette": .075 + random.random() * .05,
    }),

    "seether": lambda: extend("invert", "shadow", "symmetry", {
        "hue_range": 1.0 + random.random(),
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(4, 6),
        "post_ridges": True,
        "ridges": True,
        "speed": .05,
        "voronoi_alpha": .25 + random.random() * .25,
        "warp_range": .25,
        "warp_octaves": 6,
        "with_glowing_edges": 1,
        "with_reverb": 1,
        "with_voronoi": voronoi.color_regions,
    }),

    "seether-reflect": lambda: extend("seether", {
        "post_reflect_range": random.randint(40, 60),
    }),

    "seether-refract": lambda: extend("seether", {
        "post_refract_range": random.randint(2, 4),
    }),

    "shape-party": lambda: extend("invert", {
        "distrib": distrib.ones,
        "freq": 23,
        "mask": random_member(mask.procedural_members()),
        "point_freq": 2,
        "posterize_levels": 1,
        "rgb": True,
        "spline_order": interp.cosine,
        "voronoi_metric": distance.manhattan,
        "voronoi_nth": 1,
        "voronoi_refract": .125 + random.random() * .25,
        "with_aberration": .075 + random.random() * .075,
        "with_bloom": .075 + random.random() * .075,
        "with_voronoi": voronoi.range,
    }),

    "shatter": lambda: extend("basic", "maybe-invert", "outline", {
        "point_freq": random.randint(3, 5),
        "post_refract_range": 1.0 + random.random(),
        "post_refract_y_from_offset": True,
        "posterize_levels": random.randint(4, 6),
        "rgb": coin_flip(),
        "speed": .05,
        "voronoi_metric": random_member(distance.absolute_members()),
        "voronoi_inverse": coin_flip(),
        "with_voronoi": voronoi.range_regions,
    }),

    "shimmer": lambda: {
        "deriv": distance.euclidean,
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
        "posterize_levels": random.randint(1, 5),
        "rgb": coin_flip(),
        "speed": .025,
    }),

    "sideways": lambda: extend("bloom", "crt", "multires-low", "pixel-sort", "shadow", {
        "freq": random.randint(6, 12),
        "distrib": distrib.ones,
        "mask": mask.script,
        "reflect_range": 5.0,
        "saturation": .06125 + random.random() * .125,
        "sin": random.random() * 4,
        "spline_order": random_member([m for m in interp if m != interp.constant]),
    }),

    "sine-here-please": lambda: extend("multires", "shadow", {
        "sin": 25 + random.random() * 200,
    }),

    "sined-multifractal": lambda: extend("bloom", "multires-ridged", {
        "distrib": distrib.uniform,
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
        "freq": random.randint(2, 3),
        "hue_range": .5,
        "point_freq": random.randint(1, 3),
        "post_reindex_range": .25 + random.random() * .333,
        "reindex_range": .5 + random.random() * .666,
        "ripple_range": .0125 + random.random() * .016667,
        "speed": .025,
        "voronoi_alpha": .5 + random.random() * .333,
        "voronoi_refract": random.randint(3, 5) * .2,
        "voronoi_refract_y_from_offset": True,
        "warp_range": .0375 + random.random() * .0375,
        "with_voronoi": voronoi.color_range,
    },

    "smoke-on-the-water": lambda: extend("bloom", "dither", "shadows", {
        "octaves": 8,
        "point_freq": 10,
        "post_saturation": .5,
        "ridges": 1,
        "voronoi_alpha": .5,
        "voronoi_inverse": True,
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
        "with_worms": worms.random,
        "worms_density": 1000,
    }),

    "soft-cells": lambda: extend("bloom", {
        "point_distrib": random_member(point, mask.nonprocedural_members()),
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
        "rgb": coin_flip(),
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

    "soup": lambda: extend("bloom", "density-map", "shadow", {
        "freq": random.randint(2, 3),
        "point_freq": random.randint(2, 3),
        "post_refract_range": .5 + random.random() * .25,
        "post_refract_y_from_offset": True,
        "voronoi_inverse": True,
        "with_voronoi": voronoi.flow,
        "with_worms": worms.random,
        "worms_alpha": .875 + random.random() * .125,
        "worms_density": 500,
        "worms_kink": 4.0 + random.random() * 2.0,
    }),

    "spaghettification": lambda: extend("aberration", "bloom", "density-map", "multires-low", "shadow", {
        "point_freq": 1,
        "voronoi_inverse": True,
        "warp_range": .5 + random.random() * .5,
        "with_palette": None,
        "with_voronoi": voronoi.flow,
        "with_worms": worms.chaotic,
        "worms_alpha": .75,
        "worms_density": 1500,
        "worms_stride": random.randint(150, 350),
    }),

    "spectrogram": lambda: extend("dither", "filthy", {
        "distrib": distrib.row_index,
        "freq": random.randint(256, 512),
        "hue_range": .5 + random.random() * .5,
        "mask": mask.bar_code,
        "spline_order": interp.constant,
    }),

    "spiral-clouds": lambda: extend("multires", "shadow", "wormhole", {
        "lattice_drift": 1.0,
        "saturation-distrib": distrib.ones,
        "wormhole_alpha": .333 + random.random() * .333,
        "wormhole_stride": .001 + random.random() * .0005,
        "wormhole_kink": random.randint(40, 50),
    }),

    "spiral-in-spiral": lambda: {
        "point_distrib": point.spiral if coin_flip() else point.rotating,
        "point_freq": 10,
        "reverb_iterations": random.randint(1, 4),
        "with_reverb": random.randint(0, 6),
        "with_voronoi": random_member([voronoi.range, voronoi.color_range]),
        "with_worms": random_member([m for m in worms.all() if m != worms.random]),
        "worms_density": 500,
        "worms_duration": 1,
        "worms_kink": random.randint(5, 25),
    },

    "spiraltown": lambda: extend("wormhole", {
        "freq": 2,
        "hue_range": 1,
        "reflect_range": random.randint(3, 6) * .5,
        "spline_order": random_member([m for m in interp if m != interp.constant]),
        "wormhole_kink": random.randint(5, 20),
        "wormhole_stride": random.random() * .05,
    }),

    "splash": lambda: extend("aberration", "bloom", "tint", "vaseline", {
        "distrib": distrib.ones,
        "freq": 3,
        "lattice_drift": 1,
        "mask": mask.dropout,
        "octave_blending": blend.reduce_max,
        "octaves": 6,
        "post_deriv": distance.chebyshev,
        "rgb": True,
        "spline_order": interp.bicubic,
    }),

    "splork": lambda: {
        "distrib": distrib.ones,
        "freq": 33,
        "mask": mask.bank_ocr,
        "point_freq": 2,
        "posterize_levels": 1,
        "rgb": True,
        "spline_order": interp.cosine,
        "voronoi_metric": distance.chebyshev,
        "voronoi_nth": 1,
        "voronoi_refract": .125,
        "with_voronoi": voronoi.color_range,
    },

    "square-stripes": lambda: {
        "hue_range": random.random(),
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
        "point_freq": 2,
        "point_generations": random.randint(2, 3),
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev, distance.triangular]),
        "voronoi_nth": random.randint(1, 3),
        "voronoi_refract": .73,
        "with_voronoi": voronoi.color_range,
    },

    "stackin-bricks": lambda: {
        "point_freq": 10,
        "voronoi_metric": distance.triangular,
        "voronoi_inverse": True,
        "with_voronoi": voronoi.color_range,
    },

    "star-cloud": lambda: extend("bloom", "sobel", {
        "deriv": distance.euclidean,
        "freq": 2,
        "hue_range": random.random() * 2.0,
        "point_freq": 10,
        "reflect_range": random.random() + 2.5,
        "spline_order": interp.cosine,
        "voronoi_refract": random.randint(2, 4) * .5,
        "with_voronoi": voronoi.flow,
    }),

    "starfield": lambda: extend("aberration", "dither", "bloom", "multires-low", "nebula", {
        "distrib": distrib.exp,
        "freq": random.randint(200, 300),
        "mask": mask.sparse,
        "mask_static": True,
        "post_brightness": -.333,
        "post_contrast": 3,
        "spline_order": interp.linear,
        "with_vignette": .25 + random.random() * .25,
    }),

    "stepper": lambda: extend("voronoi", "symmetry", "outline", {
        "hue_range": random.random(),
        "saturation": random.random(),
        "point_freq": random.randint(5, 10),
        "point_corners": coin_flip(),
        "point_distrib": random_member(point.circular_members()),
        "voronoi_metric": random_member([distance.manhattan, distance.chebyshev, distance.triangular]),
        "voronoi_nth": random.randint(0, 25),
        "with_voronoi": random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),
    }),

    "string-theory": lambda: extend("multires-low", "lens", {
        "erosion_worms_alpha": .875 + random.random() * .125,
        "erosion_worms_contraction": 4.0 + random.random() * 2.0,
        "erosion_worms_density": .25 + random.random() * .125,
        "erosion_worms_iterations": random.randint(1250, 2500),
        "octaves": random.randint(2, 4),
        "rgb": True,
        "ridges": False,
        "with_bloom": .125 + random.random() * .0625,
        "with_erosion_worms": True,
        "with_palette": None,
    }),

    "subpixelator": lambda: extend("basic", "funhouse", "subpixels"),

    "symmetry": lambda: extend("maybe-palette", {
        "corners": True,
        "freq": 2,
    }),

    "symmetry-lowpoly": lambda: extend("lowpoly", "symmetry", {
        "lowpoly_distrib": random_member(point.circular_members()),
        "lowpoly_freq": random.randint(4, 15),
    }),

    "teh-matrex-haz-u": lambda: extend("bloom", "crt", {
        "distrib": distrib.exp,
        "freq": (random.randint(2, 4), random.randint(48, 96)),
        "glyph_map_zoom": random.randint(2, 4),
        "hue_rotation": .4 + random.random() * .2,
        "hue_range": .25,
        "lattice_drift": 1,
        "mask": mask.sparse,
        "post_saturation": 2,
        "spline_order": interp.linear,
        "with_glyph_map": random_member([
            random_member([mask.alphanum_binary, mask.alphanum_numeric, mask.alphanum_hex]),
            mask.truetype,
            mask.ideogram,
            mask.invaders_square,
            random_member([mask.fat_lcd, mask.fat_lcd_binary, mask.fat_lcd_numeric, mask.fat_lcd_hex]),
            mask.emoji,
        ]),
    }),

    "tensorflower": lambda: extend("bloom", "symmetry", {
        "hue_range": random.random(),
        "point_corners": True,
        "point_distrib": random_member([point.square, point.h_hex, point.v_hex]),
        "point_freq": 2,
        "rgb": coin_flip(),
        "spline_order": interp.constant,
        "vortex_range": random.randint(8, 25),
        "with_voronoi": voronoi.range_regions,
    }),

    "terra-terribili": lambda: extend("multires-ridged", "palette", "shadow", {
        "hue_range": .5 + random.random() * .5,
        "lattice_drift": 1.0,
    }),

    "the-arecibo-response": lambda: extend("snow", "value-mask", {
        "freq": random.randint(42, 210),
        "mask": mask.arecibo,
    }),

    "the-data-must-flow": lambda: extend("bloom", {
        "freq": 2,
        "post_contrast": 2,
        "post_deriv": distance.euclidean,
        "rgb": True,
        "with_worms": worms.obedient,
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
        "voronoi_metric": random_member(distance.all()),
        "with_voronoi": voronoi.color_range,
        "with_worms": random_member(worms.all()),
        "worms_alpha": 1,
        "worms_duration": random.randint(1, 4),
        "worms_density": 500,
        "worms_kink": random.randint(6, 24),
    },

    "time-to-reflect": lambda: extend("shadow", "symmetry", {
        "post_reflect_range": 5.0,
        "post_ridges": True,
        "reflect_range": random.randint(35, 70),
        "ridges": True,
    }),

    "timeworms": lambda: extend("bloom", "density-map", {
        "freq": random.randint(8, 36),
        "hue_range": 0,
        "mask": mask.sparse,
        "mask_static": True,
        "octaves": random.randint(1, 3),
        "reflect_range": random.randint(0, 1) * random.random() * 2,
        "spline_order": random_member([m for m in interp if m != interp.bicubic]),
        "with_worms": worms.obedient,
        "worms_alpha": 1,
        "worms_density": .25,
        "worms_duration": 10,
        "worms_stride": 2,
        "worms_kink": .25 + random.random() * 2.5,
    }),

    "traceroute": lambda: extend("multires", {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(2, 6),
        "mask": random_member(mask),
        "with_worms": random_member([worms.obedient, worms.crosshatch, worms.unruly]),
        "worms_density": 500,
        "worms_kink": random.randint(5, 25),
    }),

    "tri-hard": lambda: {
        "hue_range": .125 + random.random(),
        "point_freq": random.randint(8, 10),
        "posterize_levels": 6,
        "voronoi_alpha": .333 + random.random() * .333,
        "voronoi_metric": random_member([distance.octagram, distance.triangular, distance.hexagram]),
        "voronoi_refract": .333 + random.random() * .333,
        "voronoi_refract_y_from_offset": False,
        "with_outline": distance.euclidean,
        "with_voronoi": voronoi.color_range,
    },

    "triangular": lambda: extend("multires", "sobel", {
        "corners": True,
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "freq": random.randint(1, 4) * 2,
        "mask": random_member([mask.h_tri, mask.v_tri]),
    }),

    "tribbles": lambda: extend("bloom", "invert", {
        "freq": random.randint(4, 10),
        "hue_rotation": 0.375 + random.random() * .15,
        "hue_range": 0.125 + random.random() * .125,
        "saturation": .375 + random.random() * .15,
        "octaves": 3,
        "point_distrib": point.h_hex,
        "point_freq": random.randint(2, 5) * 2,
        "ridges": True,
        "voronoi_alpha": 0.5 + random.random() * .25,
        "warp_freq": random.randint(2, 4),
        "warp_octaves": random.randint(2, 4),
        "warp_range": 0.025 + random.random() * .005,
        "with_voronoi": voronoi.range_regions,
        "with_worms": worms.unruly,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 750,
        "worms_duration": .5,
        "worms_stride_deviation": .5,
    }),

    "triblets": lambda: extend("bloom", "multires", {
        "distrib": distrib.uniform,
        "freq": random.randint(3, 15) * 2,
        "mask": random_member(mask),
        "hue_rotation": 0.875 + random.random() * .15,
        "saturation": .375 + random.random() * .15,
        "warp_octaves": random.randint(1, 2),
        "warp_freq": random.randint(2, 3),
        "warp_range": 0.025 + random.random() * .05,
        "with_worms": worms.unruly,
        "worms_alpha": .875 + random.random() * .125,
        "worms_density": 750,
        "worms_duration": .5,
        "worms_stride": .5,
        "worms_stride_deviation": .25,
    }),

    "trominos": lambda: extend("bloom", "crt", "sobel", "value-mask", {
        "freq": 4 * random.randint(25, 50),
        "mask": mask.tromino,
        "spline_order": interp.constant,
    }),

    "truchet-maze": lambda: extend("value-mask", {
        "angle": random_member([0, 45]),
        "corners": True,
        "freq": 6 * random.randint(50, 100),
        "mask": random_member([mask.truchet_lines, mask.truchet_curves]),
    }),

    "truffula-spores": lambda: extend("multires-alpha", {
        "with_worms": worms.obedient,
    }),

    "twister": lambda: extend("wormhole", {
        "freq": random.randint(12, 24),
        "octaves": 2,
        "wormhole_kink": 1 + random.random() * 3,
        "wormhole_stride": .0333 + random.random() * .0333,
    }),

    "unicorn-puddle": lambda: extend("bloom", "invert", "multires", "random-hue", "shadow", {
        "distrib": distrib.uniform,
        "freq": random.randint(8, 12),
        "hue_range": 2.5,
        "lattice_drift": 1,
        "post_contrast": 1.5,
        "reflect_range": .125 + random.random() * .075,
        "ripple_freq": [random.randint(12, 64), random.randint(12, 64)],
        "ripple_kink": .5 + random.random() * .25,
        "ripple_range": .125 + random.random() * .0625,
        "with_light_leak": .5 + random.random() * .25,
    }),

    "unmasked": lambda: {
        "distrib": distrib.uniform,
        "freq": random.randint(15, 30),
        "mask": random_member(mask.procedural_members()),
        "octaves": random.randint(1, 2),
        "post_reindex_range": 1 + random.random() * 1.5,
        "with_sobel": coin_flip(),
    },

    "value-mask": lambda: extend("maybe-palette", {
        "distrib": distrib.ones,
        "mask": stash('value-mask-mask', random_member(mask)),
        "freq": [int(i * stash("value-mask-repeat", random.randint(2, 8)))
            for i in masks.mask_shape(stash("value-mask-mask"))[0:2]],
        "spline_order": random_member([m for m in interp if m != interp.bicubic]),
    }),

    "vectoroids": lambda: extend("crt", {
        "freq": 25,
        "distrib": distrib.ones,
        "mask": mask.sparse,
        "mask_static": True,
        "point_freq": 10,
        "point_drift": .25 + random.random() * .75,
        "post_deriv": distance.euclidean,
        "spline_order": interp.constant,
        "with_voronoi": voronoi.color_regions,
    }),

    "velcro": lambda: extend("wormhole", {
        "freq": 2,
        "hue_range": random.randint(0, 3),
        "octaves": random.randint(1, 2),
        "reflect_range": random.randint(6, 8) * .5,
        "spline_order": random_member([interp.cosine, interp.bicubic]),
        "wormhole_stride": random.random() * .0125,
    }),

    "vortex-checkers": lambda: extend("outline", {
        "freq": random.randint(4, 10) * 2,
        "distrib": random_member([distrib.ones, distrib.uniform]),
        "mask": mask.chess,
        "hue_range": random.random(),
        "saturation": random.random(),
        "posterize": random.randint(10, 15),
        "reverb_iterations": random.randint(2, 4),
        "sin": .5 + random.random(),
        "spline_order": interp.constant,
        "vortex_range": 2.5 + random.random() * 5,
        "with_reverb": random.randint(3, 5),
    }),

    "wall-art": lambda: extend("glyphic", "lowpoly", "rotate", {
        "lowpoly_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
    }),

    "warped-cells": lambda: extend("invert", {
        "point_distrib": random_member(point, mask.nonprocedural_members()),
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
        "spline_order": interp.constant,
        "warp_interp": random_member([m for m in interp if m != interp.constant]),
        "warp_freq": random.randint(2, 4),
        "warp_range": .25 + random.random() * .75,
        "warp_octaves": 1,
        "with_palette": None,
    }),

    "watercolor": lambda: {
        "post_saturation": .333,
        "warp_range": .5,
        "warp_octaves": 8,
        "with_fibers": True,
        "with_texture": True,
    },

    "what-do-they-want": lambda: {
        "corners": True,
        "distrib": distrib.ones,
        "freq": random.randint(1, 2) * 6,
        "octave_blending": blend.alpha,
        "octaves": 4,
        "mask": random_member([mask.invaders_square, mask.matrix]),
        "with_sobel": distance.triangular,
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

    "wild-hair": lambda: extend("multires", "erosion-worms", "shadow", "voronoi", {
        "erosion_worms_density": 25,
        "erosion_worms_alpha": .125 + random.random() * .125,
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(5, 10),
        "saturation": 0,
        "voronoi_alpha": .5 + random.random() * .5,
        "voronoi_nth": 1,
        "with_voronoi": voronoi.range,
    }),

    "wild-kingdom": lambda: extend("bloom", "dither", "maybe-invert", "outline", "random-hue", {
        "freq": 25,
        "lattice_drift": 1,
        "mask": mask.sparser,
        "mask_static": True,
        "posterize_levels": 3,
        "rgb": True,
        "ridges": True,
        "spline_order": interp.cosine,
        "warp_octaves": 2,
        "warp_range": .025,
    }),

    "wireframe": lambda: extend("basic", "bloom", "multires-low", "sobel", {
        "hue_range": random.random(),
        "saturation": random.random(),
        "lattice_drift": random.random(),
        "point_distrib": random_member(point.grid_members(), mask.nonprocedural_members()),
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
        "point_distrib": random_member(point.circular_members()),
        "point_freq": random.randint(2, 3),
        "point_generations": 2,
        "reverb_iterations": random.randint(1, 2),
        "refract_range": random.randint(0, 1) * random.random(),
        "voronoi_metric": distance.chebyshev,
        "voronoi_nth": random.randint(1, 3),
        "voronoi_alpha": .5 + random.random() * .5,
        "with_reverb": random.randint(0, 2),
        "with_voronoi": voronoi.color_range,
        "with_worms": worms.chaotic,
        "worms_alpha": .75 + random.random() * .25,
        "worms_density": 250 + random.random() * 250,
        "worms_duration": 1 + random.random() * 1.5,
        "worms_kink": 5 + random.random() * 2.0,
        "worms_stride": 2.5,
        "worms_stride_deviation": 1.25,
    },

    "wormstep": lambda: extend("basic", "bloom", {
        "corners": True,
        "lattice_drift": coin_flip(),
        "octaves": random.randint(1, 3),
        "with_worms": worms.chaotic,
        "worms_alpha": .5 + random.random() * .5,
        "worms_density": 500,
        "worms_kink": 1.0 + random.random() * 4.0,
        "worms_stride": 8.0 + random.random() * 4.0,
        "with_palette": None,
    }),

}


# Call after setting seed
def bake_presets():
    global EFFECTS_PRESETS
    EFFECTS_PRESETS = _EFFECTS_PRESETS()

    global PRESETS
    PRESETS = _PRESETS()


def extend(*args):
    args = deque(args)

    settings = {}

    settings['with_convolve'] = set()
    settings['tags'] = set()

    while args:
        arg = args.popleft()

        if isinstance(arg, str):
            these_settings = preset(arg)
            settings['tags'].add(arg)

        else:
            these_settings = arg

        settings['tags'].update(these_settings.pop('tags', set()))
        settings['with_convolve'].update(these_settings.pop('with_convolve', set()))

        settings.update(these_settings)

    del(settings['name'])  # Let the preset() function populate the resolved name

    settings['with_convolve'] = list(settings['with_convolve'])  # Convert to a JSON-friendly type
    settings['tags'] = sorted(settings['tags'])

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


def preset_by_tag(tag):
    presets = []

    for name in list(PRESETS) + list(EFFECTS_PRESETS):
        if tag in preset(name).get('tags', set()):
            presets.append(name)

    if not presets:
        return None

    return presets[random.randint(0, len(presets) - 1)]


bake_presets()
