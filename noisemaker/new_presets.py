import functools
import random

from noisemaker.composer import Effect, Preset
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
from noisemaker.presets import coin_flip, enum_range, random_member, stash

import noisemaker.masks as masks

#: A dictionary of presets for use with the artmaker-new script.
PRESETS = {
    "1969": {
        "layers": ["symmetry", "voronoi", "posterize-outline", "distressed"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "palette_name": None,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.circular,
            "voronoi_point_freq": random.randint(3, 5) * 2,
            "voronoi_nth": random.randint(1, 3),
        },
        "generator": lambda settings: {
            "rgb": True,
        },
    },

    "1976": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": 2,
        },
        "post": lambda settings: [
            Preset("dither"),
            Effect("adjust_saturation", amount=.25 + random.random() * .125)
        ]
    },

    "1985": {
        "layers": ["reindex-post", "voronoi"],
        "settings": lambda: {
            "dist_metric": distance.chebyshev,
            "reindex_range": .2 + random.random() * .1,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.random,
            "voronoi_refract": .2 + random.random() * .1
        },
        "generator": lambda settings: {
            "freq": random.randint(10, 15),
            "spline_order": interp.constant
        },
        "post": lambda settings: [
            Effect("palette", name="neon"),
            Preset("random-hue"),
            Effect("spatter"),
            Preset("be-kind-rewind")
        ]
    },

    "2001": {
        "layers": ["analog-glitch", "invert", "posterize", "vignette-bright", "aberration"],
        "settings": lambda: {
            "mask": mask.bank_ocr,
            "mask_repeat": random.randint(9, 12),
            "vignette_alpha": .75 + random.random() * .25,
            "posterize_levels": random.randint(1, 2),
        },
        "generator": lambda settings: {
            "spline_order": interp.cosine,
        }
    },

    "2d-chess": {
        "layers": ["value-mask", "voronoi"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "voronoi_alpha": 0.5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range if coin_flip() \
                else random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),  # noqa E131
            "voronoi_nth": random.randint(0, 1) * random.randint(0, 63),
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.square,
            "voronoi_point_freq": 8,
        },
        "generator": lambda settings: {
            "corners": True,
            "freq": 8,
            "mask": mask.chess,
            "spline_order": interp.constant,
        }
    },

    "aberration": {
        "post": lambda settings: [Effect("aberration", displacement=.0125 + random.random() * .006125)]
    },

    "abyssal-echoes": {
        "layers": ["multires-alpha", "desaturate", "random-hue"],
        "generator": lambda settings: {
            "rgb": True,
        },
        "octaves": lambda settings: [
            Effect("refract",
                   displacement=random.randint(20, 30),
                   from_derivative=True,
                   y_from_offset=False)
        ],
    },

    "acid": {
        "layers": ["reindex-post", "normalize"],
        "settings": lambda: {
            "reindex_range": 1.25 + random.random() * 1.25,
        },
        "generator": lambda settings: {
            "freq": random.randint(10, 15),
            "octaves": 8,
            "rgb": True,
        },
    },

    "acid-droplets": {
        "layers": ["multires", "reflect-octaves", "density-map", "random-hue", "bloom", "shadow", "desaturate"],
        "settings": lambda: {
            "palette_name": None,
            "reflect_range": 7.5 + random.random() * 3.5
        },
        "generator": lambda settings: {
            "freq": random.randint(10, 15),
            "hue_range": 0,
            "lattice_drift": 1.0,
            "mask": mask.sparse,
            "mask_static": True,
        },
    },

    "acid-grid": {
        "layers": ["voronoid", "sobel", "funhouse", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": 4,
            "voronoi_point_generations": 2,
        },
        "generator": lambda settings: {
            "lattice_drift": coin_flip(),
        },
    },

    "acid-wash": {
        "layers": ["funhouse"],
        "settings": lambda: {
            "warp_octaves": 8,
        },
        "generator": lambda settings: {
            "freq": random.randint(4, 6),
            "hue_range": 1.0,
            "ridges": True,
        },
        "post": lambda settings: [Effect("ridge"), Preset("shadow"), Preset("desaturate")]
    },

    "activation-signal": {
        "layers": ["value-mask", "maybe-palette", "glitchin-out"],
        "generator": lambda settings: {
            "freq": 4,
            "mask": mask.white_bear,
            "rgb": coin_flip(),
            "spline_order": interp.constant,
        }
    },

    "aesthetic": {
        "layers": ["maybe-derivative-post", "spatter", "maybe-invert", "be-kind-rewind", "spatter"],
        "generator": lambda settings: {
            "corners": True,
            "distrib": random_member([distrib.column_index, distrib.ones, distrib.row_index]),
            "freq": random.randint(3, 5) * 2,
            "mask": mask.chess,
            "spline_order": interp.constant,
        },
    },

    "alien-terrain-multires": {
        "layers": ["multires-ridged", "derivative-octaves", "maybe-invert", "bloom", "shadow", "desaturate"],
        "settings": lambda: {
            "deriv_alpha": .333 + random.random() * .333,
            "dist_metric": distance.euclidean,
            "palette_name": None,
        },
        "generator": lambda settings: {
            "freq": random.randint(5, 9),
            "lattice_drift": 1.0,
            "octaves": 8,
        }
    },

    "alien-terrain-worms": {
        "layers": ["multires-ridged", "invert", "voronoi", "derivative-octaves", "invert",
                   "erosion-worms", "bloom", "shadow", "dither", "contrast", "desaturate"],
        "settings": lambda: {
            "deriv_alpha": .25 + random.random() * .125,
            "dist_metric": distance.euclidean,
            "erosion_worms_alpha": .05 + random.random() * .025,
            "erosion_worms_density": random.randint(150, 200),
            "erosion_worms_inverse": True,
            "erosion_worms_xy_blend": .333 + random.random() * .16667,
            "palette_name": None,
            "voronoi_alpha": .5 + random.random() * .25,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": 10,
            "voronoi_point_distrib": point.random,
            "voronoi_refract": .25 + random.random() * .125,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 5),
            "hue_rotation": .875,
            "hue_range": .25 + random.random() * .25,
        },
    },

    "alien-transmission": {
        "layers": ["analog-glitch", "sobel", "glitchin-out"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
        }
    },

    "analog-glitch": {
        "layers": ["value-mask"],
        "settings": lambda: {
            "mask": random_member([mask.alphanum_hex, mask.lcd, mask.fat_lcd]),
            "mask_repeat": random.randint(20, 30),
        },
        "generator": lambda settings: {
            # offset by i * .5 for glitched texture lookup
            "freq": [int(i * .5 + i * settings["mask_repeat"]) for i in masks.mask_shape(settings["mask"])[0:2]],
            "mask": settings["mask"],
        }
    },

    "arcade-carpet": {
        "layers": ["basic", "funhouse", "posterize", "nudge-hue", "contrast", "dither"],
        "settings": lambda: {
            "palette_name": None,
            "posterize_levels": 3,
            "warp_freq": random.randint(75, 125),
            "warp_range": .02 + random.random() * .01,
            "warp_octaves": 1,
        },
        "generator": lambda settings: {
            "distrib": distrib.exp,
            "freq": settings["warp_freq"],
            "hue_range": 1,
            "mask": mask.sparser,
            "mask_static": True,
            "rgb": True,
        }
    },

    "are-you-human": {
        "layers": ["multires", "value-mask", "funhouse", "density-map", "desaturate", "maybe-invert", "aberration", "snow"],
        "generator": lambda settings: {
            "freq": 15,
            "hue_range": random.random() * .25,
            "hue_rotation": random.random(),
            "mask": mask.truetype,
        },
    },

    "aztec-waffles": {
        "layers": ["symmetry", "voronoi", "maybe-invert", "reflect-post"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "reflect_range": random.randint(12, 25),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(2, 4),
            "voronoi_point_distrib": point.circular,
            "voronoi_point_freq": 4,
            "voronoi_point_generations": 2,
        },
    },

    "basic": {
        "layers": ["maybe-palette", "normalize"],
        "generator": lambda settings: {
            "freq": [random.randint(2, 4), random.randint(2, 4)],
        },
    },

    "basic-lowpoly": {
        "layers": ["basic", "lowpoly"],
    },

    "basic-voronoi": {
        "layers": ["basic", "voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": random_member([voronoi.color_range, voronoi.color_regions,
                                                   voronoi.range_regions, voronoi.color_flow])
        }
    },

    "basic-voronoi-refract": {
        "layers": ["basic", "voronoi"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": 0,
            "voronoi_refract": 1.0 + random.random() * .5,
        },
        "generator": lambda settings: {
            "hue_range": .25 + random.random() * .5,
        },
    },

    "band-together": {
        "layers": ["reindex-post", "funhouse", "shadow", "normalize"],
        "settings": lambda: {
            "reindex_range": random.randint(8, 12),
            "warp_range": .333 + random.random() * .16667,
            "warp_octaves": 8,
            "warp_freq": random.randint(2, 3),
        },
        "generator": lambda settings: {
            "freq": random.randint(6, 12),
        },
    },

    "be-kind-rewind": {
        "post": lambda settings: [Effect("vhs"), Preset("crt")]
    },

    "beneath-the-surface": {
        "layers": ["multires-alpha", "reflect-octaves", "bloom", "shadow"],
        "settings": lambda: {
            "reflect_range": 10.0 + random.random() * 5.0,
        },
        "generator": lambda settings: {
            "freq": 3,
            "hue_range": 2.0 + random.random() * 2.0,
            "octaves": 5,
            "ridges": True,
        },
    },

    "benny-lava": {
        "layers": ["posterize", "maybe-palette", "funhouse", "distressed"],
        "settings": lambda: {
            "posterize_levels": 1,
            "warp_range": 1 + random.random() * .5,
        },
        "generator": lambda settings: {
            "distrib": distrib.column_index,
        },
    },

    "berkeley": {
        "layers": ["multires-ridged", "reindex-octaves", "sine-octaves"],
        "settings": lambda: {
            "palette_name": None,
            "reindex_range": .75 + random.random() * .25,
            "sine_range": 2.0 + random.random() * 2.0,
        },
        "generator": lambda settings: {
            "freq": random.randint(12, 16)
        },
        "post": lambda settings: [
            Effect("ridge"),
            Preset("shadow"),
            Preset("contrast"),
        ]
    },

    "big-data-startup": {
        "layers": ["glyphic", "dither", "desaturate"],
        "settings": lambda: {
            "posterize_levels": random.randint(2, 4),
        },
        "generator": lambda settings: {
            "mask": mask.script,
            "hue_rotation": random.random(),
            "hue_range": .0625 + random.random() * .5,
            "saturation": 1.0,
        }
    },

    "bit-by-bit": {
        "layers": ["value-mask", "bloom", "crt"],
        "settings": lambda: {
            "mask": random_member([mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric]),
            "mask_repeat": random.randint(30, 60)
        }
    },

    "bitmask": {
        "layers": ["multires-low", "value-mask", "bloom"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
            "mask_repeat": random.randint(7, 15),
        },
        "generator": lambda settings: {
            "ridges": True,
        }
    },

    "blacklight-fantasy": {
        "layers": ["voronoi", "funhouse", "posterize", "sobel", "invert", "bloom", "dither", "nudge-hue"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "posterize_levels": 3,
            "voronoi_refract": .5 + random.random() * 1.25,
            "warp_octaves": random.randint(1, 4),
            "warp_range": random.randint(0, 1) * random.random(),
        },
        "generator": lambda settings: {
            "rgb": True,
        },
    },

    "blockchain-stock-photo-background": {
        "layers": ["value-mask", "glitchin-out", "rotate", "vignette-dark"],
        "settings": lambda: {
            "angle": random.randint(5, 35),
            "vignette_alpha": 1.0,
            "mask": random_member([mask.alphanum_binary, mask.alphanum_hex,
                                   mask.alphanum_numeric, mask.bank_ocr]),
            "mask_repeat": random.randint(20, 40),
        },
    },

    "bloom": {
        "layers": ["normalize"],
        "settings": lambda: {
            "bloom_alpha": .25 + random.random() * .125,
        },
        "post": lambda settings: [
            Effect("bloom", alpha=settings["bloom_alpha"])
        ]
    },

    "blotto": {
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "rgb": coin_flip(),
        },
        "post": lambda settings: [Effect("spatter", color=False), Preset("maybe-palette")]
    },

    "branemelt": {
        "layers": ["multires", "sine-octaves", "reflect-post", "bloom", "shadow", "brightness", "contrast"],
        "settings": lambda: {
            "brightness": .125,
            "contrast": 1.5,
            "palette_name": None,
            "reflect_range": .0333 + random.random() * .016667,
            "shadow_alpha": .666 + random.random() * .333,
            "sine_range": random.randint(64, 96),
        },
        "generator": lambda settings: {
            "freq": random.randint(12, 24),
        },
    },

    "branewaves": {
        "layers": ["value-mask", "ripples", "bloom"],
        "settings": lambda: {
            "mask": random_member(mask.grid_members()),
            "mask_repeat": random.randint(5, 10),
            "ripple_freq": 2,
            "ripple_kink": 1.5 + random.random() * 2,
            "ripple_range": .15 + random.random() * .15,
        },
        "generator": lambda settings: {
            "ridges": True,
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        },
    },

    "brightness": {
        "settings": lambda: {
            "brightness": .125 + random.random() * .0625
        },
        "post": lambda settings: [Effect("adjust_brightness", amount=settings["brightness"])]
    },

    "bringing-hexy-back": {
        "layers": ["voronoi", "funhouse", "maybe-invert", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.v_hex if coin_flip() else point.h_hex,
            "voronoi_point_freq": random.randint(4, 7) * 2,
            "warp_range": .05 + random.random() * .25,
            "warp_octaves": random.randint(1, 4),
        },
        "generator": lambda settings: {
            "freq": settings["voronoi_point_freq"],
            "hue_range": .25 + random.random() * .75,
            "rgb": coin_flip(),
        }
    },

    "broken": {
        "layers": ["multires-low", "reindex-octaves", "posterize", "glowing-edges", "dither", "desaturate"],
        "settings": lambda: {
            "posterize_levels": 3,
            "reindex_range": random.randint(3, 4),
            "speed": .025,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 4),
            "lattice_drift": 2,
            "rgb": True,
        },
    },

    "bubble-chamber": {
        "layers": ["basic", "worms", "brightness", "contrast", "glowing-edges", "tint",
                   "bloom", "snow", "lens"],
        "settings": lambda: {
            "brightness": .125,
            "contrast": 2.5 + random.random() * 1.25,
            "palette_name": None,
            "vignette_alpha": .25 + random.random() * .125,
            "worms_alpha": .925,
            "worms_behavior": worms.chaotic,
            "worms_density": .25 + random.random() * .125,
            "worms_drunken_spin": True,
            "worms_drunkenness": .125 + random.random() * .06125,
            "worms_stride_deviation": 5.0 + random.random() * 2.5,
        },
        "generator": lambda settings: {
            "hue_range": 1.0,
            "distrib": distrib.exp,
        },
    },

    "bubble-machine": {
        "layers": ["posterize", "wormhole", "reverb", "outline", "maybe-invert"],
        "settings": lambda: {
            "posterize_levels": random.randint(8, 16),
            "reverb_iterations": random.randint(1, 3),
            "reverb_octaves": random.randint(3, 5),
            "wormhole_stride": .1 + random.random() * .05,
            "wormhole_kink": .5 + random.random() * 4,
        },
        "generator": lambda settings: {
            "corners": True,
            "distrib": distrib.uniform,
            "freq": random.randint(3, 6) * 2,
            "mask": random_member([mask.h_hex, mask.v_hex]),
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        }
    },

    "bubble-multiverse": {
        "layers": ["voronoi", "refract-post", "density-map", "random-hue", "bloom", "shadow"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "refract_range": .125 + random.random() * .05,
            "speed": .05,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": 10,
            "voronoi_refract": .625 + random.random() * .25,
        },
    },

    "carpet": {
        "layers": ["worms", "grime"],
        "settings": lambda: {
            "worms_alpha": .25 + random.random() * .25,
            "worms_behavior": worms.chaotic,
            "worms_stride": .333 + random.random() * .333,
            "worms_stride_deviation": .25
        },
    },

    "celebrate": {
        "layers": ["posterize", "maybe-palette", "distressed"],
        "settings": lambda: {
            "posterize_levels": random.randint(3, 5),
            "speed": .025,
        },
        "generator": lambda settings: {
            "brightness_distrib": distrib.ones,
            "hue_range": 1,
        }
    },

    "cell-reflect": {
        "layers": ["voronoi", "reflect-post", "derivative-post", "density-map", "maybe-invert",
                   "bloom", "dither", "desaturate"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "palette_name": None,
            "reflect_range": random.randint(2, 4) * 5,
            "saturation": .5 + random.random() * .25,
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": coin_flip(),
            "voronoi_point_distrib": random_member([m for m in point if m not in point.grid_members()]),
            "voronoi_point_freq": random.randint(2, 3),
        }
    },

    "cell-refract": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "voronoi_diagram_type": voronoi.range,
            "voronoi_point_freq": random.randint(3, 4),
            "voronoi_refract": random.randint(8, 12) * .5,
        },
        "generator": lambda settings: {
            "rgb": coin_flip(),
            "ridges": True,
        },
        "post": lambda settings: [
            Effect("ridge"),
        ]
    },

    "cell-refract-2": {
        "layers": ["voronoi", "refract-post", "derivative-post", "density-map", "desaturate"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "refract_range": random.randint(1, 3) * .25,
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member([m for m in point if m not in point.grid_members()]),
            "voronoi_point_freq": random.randint(2, 3),
        }
    },

    "cell-worms": {
        "layers": ["multires-low", "voronoi", "worms", "density-map", "random-hue", "desaturate"],
        "settings": lambda: {
            "voronoi_alpha": .75,
            "voronoi_point_distrib": random_member(point, mask.nonprocedural_members()),
            "voronoi_point_freq": random.randint(2, 4),
            "worms_density": 1500,
            "worms_kink": random.randint(16, 32),
            "worms_stride_deviation": 0,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 7),
            "hue_range": .125 + random.random() * .875,
        }
    },

    "classic-desktop": {
        "layers": ["basic", "lens-warp"],
        "generator": lambda settings: {
            "hue_range": .333 + random.random() * .333,
            "lattice_drift": random.random(),
        }
    },

    "clouds": {
        "post": lambda settings: [Effect("clouds"), Preset("bloom"), Preset("dither")]
    },

    "color-flow": {
        "layers": ["basic-voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": voronoi.color_flow,
        },
        "generator": lambda settings: {
            "freq": 64,
            "hue_range": 5,
        }
    },

    "concentric": {
        "layers": ["voronoi", "contrast", "maybe-palette", "wobble"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "speed": .75,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_refract": random.randint(8, 16),
            "voronoi_point_drift": 0,
            "voronoi_point_freq": random.randint(1, 2),
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": 2,
            "mask": mask.h_bar,
            "rgb": True,
            "spline_order": interp.constant,
        }
    },

    "conference": {
        "layers": ["value-mask", "sobel"],
        "generator": lambda settings: {
            "freq": 4 * random.randint(6, 12),
            "mask": mask.halftone,
            "spline_order": interp.cosine,
        }
    },

    "contrast": {
        "settings": lambda: {
            "contrast": 1.25 + random.random() * .25
        },
        "post": lambda settings: [Effect("adjust_contrast", amount=settings["contrast"])]
    },

    "cool-water": {
        "layers": ["refract-octaves", "reflect-octaves", "ripples", "funhouse", "bloom"],
        "settings": lambda: {
            "reflect_range": .16667 + random.random() * .16667,
            "refract_range": .25 + random.random() * .125,
            "refract_y_from_offset": True,
            "ripple_range": .005 + random.random() * .0025,
            "ripple_kink": random.randint(2, 4),
            "ripple_freq": random.randint(2, 4),
            "warp_range": .0625 + random.random() * .0625,
            "warp_freq": random.randint(2, 3),
        },
        "generator": lambda settings: {
            "distrib": distrib.uniform,
            "freq": random.randint(7, 10),
            "hue_range": .05 + random.random() * .05,
            "hue_rotation": .5125 + random.random() * .025,
            "lattice_drift": 1.0,
            "octaves": 4,
        }
    },

    "corner-case": {
        "layers": ["multires-ridged", "desaturate", "dither"],
        "generator": lambda settings: {
            "corners": True,
            "lattice_drift": coin_flip(),
            "spline_order": interp.constant,
        },
    },

    "cosmic-thread": {
        "layers": ["worms", "brightness", "contrast", "bloom"],
        "settings": lambda: {
            "brightness": .1,
            "contrast": 2.5,
            "worms_alpha": .875,
            "worms_behavior": random_member(worms.all()),
            "worms_density": .125,
            "worms_drunkenness": .125 + random.random() * .25,
            "worms_duration": 125,
            "worms_kink": 1.0,
            "worms_stride": .75,
            "worms_stride_deviation": 0.0
        },
        "generator": lambda setings: {
            "rgb": True,
        },
    },

    "cobblestone": {
        "layers": ["bringing-hexy-back", "desaturate"],
        "settings": lambda: {
            "saturation": .0 + random.random() * .05,
            "shadow_alpha": 1.0,
            "voronoi_point_freq": random.randint(3, 4) * 2,
            "warp_freq": [random.randint(3, 4), random.randint(3, 4)],
            "warp_range": .125,
            "warp_octaves": 8
        },
        "generator": lambda settings: {
            "hue_range": .1 + random.random() * .05,
        },
        "post": lambda settings: [
            Effect("texture"),
            Preset("shadow", settings={"shadow_alpha": settings["shadow_alpha"]}),
            Effect("adjust_brightness", amount=-.125),
            Preset("contrast"),
            Effect("bloom", alpha=1.0)
        ],
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

    "crooked": {
        "layers": ["starfield", "pixel-sort", "glitchin-out"],
        "settings": lambda: {
            "pixel_sort_angled": True,
            "pixel_sort_darkest": False
        }
    },

    "crt": {
        "layers": ["scanline-error", "snow"],
        "post": lambda settings: [Effect("crt")]
    },

    "crystallize": {
        "layers": ["voronoi", "vignette-bright", "bloom", "desaturate"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "voronoi_point_freq": 4,
            "voronoi_alpha": .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 4,
        }
    },

    "cubert": {
        "layers": ["voronoi", "crt", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_inverse": True,
            "voronoi_point_distrib": point.h_hex,
            "voronoi_point_freq": random.randint(4, 6),
        },
        "generator": lambda settings: {
            "freq": random.randint(4, 6),
            "hue_range": .5 + random.random(),
        }
    },

    "cubic": {
        "layers": ["basic-voronoi", "outline", "bloom"],
        "settings": lambda: {
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(2, 8),
            "voronoi_point_distrib": point.concentric,
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_diagram_type": random_member([voronoi.range, voronoi.color_range]),
        }
    },

    "cyclic-dilation": {
        "layers": ["voronoi", "reindex-post"],
        "settings": lambda: {
            "reindex_range": random.randint(4, 6),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_corners": True,
        },
        "generator": lambda settings: {
            "freq": random.randint(24, 48),
            "hue_range": .25 + random.random() * 1.25,
        }
    },

    "dark-matter": {
        "layers": ["multires-alpha", "reflect-octaves"],
        "settings": lambda: {
            "reflect_range": random.randint(20, 30),
        },
        "generator": lambda settings: {
            "octaves": 5,
        },
    },

    "deadbeef": {
        "layers": ["value-mask", "corrupt", "bloom", "crt", "vignette-dark"],
        "generator": lambda settings: {
            "freq": 6 * random.randint(9, 24),
            "mask": mask.alphanum_hex,
        }
    },

    "death-star-plans": {
        "layers": ["voronoi", "refract-post", "rotate", "posterize", "sobel", "invert", "crt", "vignette-dark"],
        "settings": lambda: {
            "dist_metric": random_member([distance.chebyshev, distance.manhattan]),
            "posterize_levels": random.randint(3, 4),
            "refract_range": .5 + random.random() * .25,
            "refract_y_from_offset": True,
            "voronoi_alpha": 1,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": random.randint(1, 3),
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(2, 3),
        },
    },

    "deep-field": {
        "layers": ["multires", "refract-octaves", "funhouse"],
        "settings": lambda: {
            "palette_name": None,
            "refract_range": .2 + random.random() * .1,
            "warp_freq": [2, 2],
            "warp_range": .025,
            "warp_signed_range": True,
        },
        "generator": lambda settings: {
            "distrib": distrib.uniform,
            "freq": random.randint(8, 10),
            "hue_range": 1,
            "mask": mask.sparser,
            "mask_static": True,
            "lattice_drift": 1,
            "octave_blending": blend.reduce_max,
            "octaves": 5,
        }
    },

    "deeper": {
        "layers": ["multires-alpha"],
        "generator": lambda settings: {
            "hue_range": 1.0,
            "octaves": 8,
        }
    },

    "degauss": {
        "post": lambda settings: [
            Effect("degauss", displacement=.0625 + random.random() * .03125),
            Preset("crt"),
        ]
    },

    "density-map": {
        "post": lambda settings: [Effect("density_map"), Effect("convolve", kernel=mask.conv2d_invert), Preset("dither")]
    },

    "density-wave": {
        "layers": [random_member(["basic", "symmetry"]), "reflect-post", "density-map", "invert", "bloom"],
        "settings": lambda: {
            "reflect_range": random.randint(3, 8),
        },
        "generator": lambda settings: {
            "saturation": random.randint(0, 1),
        }
    },

    "derivative-octaves": {
        "settings": lambda: {
            "deriv_alpha": 1.0,
            "dist_metric": random_member(distance.absolute_members())
        },
        "octaves": lambda settings: [Effect("derivative", dist_metric=settings["dist_metric"], alpha=settings["deriv_alpha"])]
    },

    "derivative-post": {
        "settings": lambda: {
            "deriv_alpha": 1.0,
            "dist_metric": random_member(distance.absolute_members())
        },
        "post": lambda settings: [Effect("derivative", dist_metric=settings["dist_metric"], alpha=settings["deriv_alpha"])]
    },

    "desaturate": {
        "settings": lambda: {
            "saturation": .333 + random.random() * .16667
        },
        "post": lambda settings: [Effect("adjust_saturation", amount=settings["saturation"])]
    },

    "different": {
        "layers": ["multires", "sine-octaves", "reflect-octaves", "reindex-octaves", "funhouse"],
        "settings": lambda: {
            "reflect_range": 7.5 + random.random() * 5.0,
            "reindex_range": .25 + random.random() * .25,
            "sine_range": random.randint(15, 25),
            "speed": .025,
            "warp_range": .0375 * random.random() * .0375,
        },
        "generator": lambda settings: {
            "freq": [random.randint(8, 12), random.randint(8, 12)]
        },
    },

    "distressed": {
        "layers": ["dither", "filthy"],
        "post": lambda settings: [Preset("desaturate")]
    },

    "distance": {
        "layers": ["multires", "derivative-octaves", "bloom", "shadow", "contrast", "rotate", "lens"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "vignette_alpha": .25 + random.random() * .125,
        },
        "generator": lambda settings: {
            "freq": [random.randint(4, 5), random.randint(2, 3)],
            "distrib": distrib.exp,
            "lattice_drift": 1,
            "saturation": .06125 + random.random() * .125,
        },
    },

    "dither": {
        "settings": lambda: {
            "dither_alpha": .1 + random.random() * .05
        },
        "post": lambda settings: [Effect("dither", alpha=settings["dither_alpha"])]
    },

    # "dla-cells": {
    # extend "bloom"
    # "dla_padding": random.randint(2, 8),
    # "hue_range": random.random() * 1.5,
    # "point_distrib": random_member(point, mask.nonprocedural_members()),
    # "point_freq": random.randint(2, 8),
    # "voronoi_alpha": random.random(),
    # "with_dla": .5 + random.random() * .5,
    # "with_voronoi": random_member(voronoi),
    # },

    "dla": {
        "settings": lambda: {
            "dla_alpha": .666 + random.random() * .333,
            "dla_padding": random.randint(2, 8),
            "dla_seed_density": .2 + random.random() * .1,
            "dla_density": .1 + random.random() * .05,
        },
        "post": lambda settings: [
            Effect("dla",
                   alpha=settings["dla_alpha"],
                   padding=settings["dla_padding"],
                   seed_density=settings["dla_seed_density"],
                   density=settings["dla_density"])
        ]
    },

    "dla-forest": {
        "layers": ["dla", "reverb", "contrast", "bloom"],
        "settings": lambda: {
            "dla_padding": random.randint(2, 8),
            "reverb_iterations": random.randint(2, 4),
        }
    },

    "dmt": {
        "layers": ["voronoi", "kaleido", "refract-post", "bloom", "vignette-dark", "contrast", "normalize"],
        "settings": lambda: {
            "contrast": 2.5,
            "dist_metric": random_member(distance.absolute_members()),
            "kaleido_point_freq": 4,
            "kaleido_point_distrib": random_member([point.square, point.waffle]),
            "kaleido_sides": 4,
            "refract_range": .075 + random.random() * .075,
            "speed": .025,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_point_distrib": random_member([point.square, point.waffle]),
            "voronoi_point_freq": 4,
            "voronoi_refract": .075 + random.random() * .075,
        },
        "generator": lambda settings: {
            "brightness_distrib": random_member([distrib.ones, distrib.uniform]),
            "mask": None if coin_flip() else mask.dropout,
            "freq": 4,
            "hue_range": 2.5 + random.random() * 1.25,
        },
    },

    "domain-warp": {
        "layers": ["multires-ridged", "refract-post", "vaseline", "dither", "vignette-dark", "desaturate"],
        "settings": lambda: {
            "refract_range": .25 + random.random() * .25,
            "vignette_alpha": .333 + random.random() * .16667,
        }
    },

    "dropout": {
        "layers": ["derivative-post", "maybe-invert"],
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": [random.randint(4, 6), random.randint(2, 4)],
            "mask": mask.dropout,
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(5, 6),
            "rgb": coin_flip(),
            "spline_order": interp.constant,
        }
    },

    "eat-static": {
        "layers": ["be-kind-rewind", "scanline-error", "crt"],
        "settings": lambda: {
            "speed": 2.0,
        },
        "generator": lambda settings: {
            "distrib": distrib.fastnoise,
            "freq": 512,
            "saturation": 0,
        }
    },

    "electric-worms": {
        "layers": ["voronoi", "worms", "density-map", "glowing-edges", "bloom"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.octagram, distance.triangular]),
            "voronoi_alpha": .25 + random.random() * .25,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(0, 3),
            "voronoi_point_freq": random.randint(3, 6),
            "voronoi_point_distrib": point.random,
            "worms_alpha": .666 + random.random() * .333,
            "worms_behavior": worms.random,
            "worms_density": 1000,
            "worms_duration": 1,
            "worms_kink": random.randint(7, 9),
            "worms_stride_deviation": 16,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 6),
            "lattice_drift": 1,
        },
    },

    "emo": {
        "layers": ["value-mask", "voronoi", "contrast", "rotate", "desaturate", "tint", "lens"],
        "settings": lambda: {
            "contrast": 4.0,
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "mask": mask.emoji,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_refract": .125 + random.random() * .25,
        },
        "generator": lambda settings: {
            "spline_order": interp.cosine,
        }
    },

    "emu": {
        "layers": ["value-mask", "maybe-palette", "voronoi", "desaturate", "distressed"],
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "mask": stash("mask", random_member(enum_range(mask.emoji_00, mask.emoji_26))),
            "mask_repeat": 1,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_point_distrib": stash("mask"),
            "voronoi_refract": .125 + random.random() * .125,
            "voronoi_refract_y_from_offset": False,
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "spline_order": interp.constant,
        },
    },

    "entities": {
        "layers": ["value-mask", "refract-octaves", "normalize"],
        "settings": lambda: {
            "refract_range": .1 + random.random() * .05,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "mask": mask.invaders_square,
            "mask_repeat": random.randint(3, 4) * 2,
        },
        "generator": lambda settings: {
            "distrib": distrib.simplex,
            "hue_range": 2.0 + random.random() * 2.0,
            "spline_order": interp.cosine,
        },
    },

    "entity": {
        "layers": ["entities", "sobel", "invert", "bloom", "random-hue", "lens"],
        "settings": lambda: {
            "refract_range": .025 + random.random() * .0125,
            "refract_signed_range": True,
            "refract_y_from_offset": False,
            "speed": .05,
        },
        "generator": lambda settings: {
            "corners": True,
            "distrib": distrib.ones,
            "freq": 6,
            "hue_range": 1.0 + random.random() * .5,
        }
    },

    "erosion-worms": {
        "settings": lambda: {
            "erosion_worms_alpha": .5 + random.random() * .5,
            "erosion_worms_contraction": .5 + random.random() * .5,
            "erosion_worms_density": random.randint(25, 100),
            "erosion_worms_inverse": False,
            "erosion_worms_iterations": random.randint(25, 100),
            "erosion_worms_xy_blend": .75 + random.random() * .25
        },
        "post": lambda settings: [
            Effect("erosion_worms",
                   alpha=settings["erosion_worms_alpha"],
                   contraction=settings["erosion_worms_contraction"],
                   density=settings["erosion_worms_density"],
                   inverse=settings["erosion_worms_inverse"],
                   iterations=settings["erosion_worms_iterations"],
                   xy_blend=settings["erosion_worms_xy_blend"]),
            Effect("normalize")
        ]
    },

    "escape-velocity": {
        "layers": ["multires-low", "erosion-worms", "lens"],
        "settings": lambda: {
            "erosion_worms_contraction": .2 + random.random() * .1,
            "erosion_worms_iterations": random.randint(625, 1125),
        },
        "generator": lambda settings: {
            "distrib": random_member([distrib.lognormal, distrib.exp, distrib.uniform]),
            "rgb": coin_flip(),
        }
    },

    "explore": {
        "layers": ["dmt", "kaleido", "bloom", "contrast", "lens"],
        "settings": lambda: {
            "refract_range": .75 + random.random() * .75,
            "kaleido_sides": random.randint(3, 18),
        },
        "generator": lambda settings: {
            "hue_range": .75 + random.random() * .75,
            "brightness_distrib": None,
        }
    },

    "falsetto": {
        "post": lambda settings: [Effect("false_color")]
    },

    "fast-eddies": {
        "layers": ["basic", "voronoi", "worms", "contrast", "desaturate"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "palette_name": None,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": random.randint(2, 10),
            "voronoi_refract": 1.0,
            "worms_alpha": .5 + random.random() * .5,
            "worms_behavior": worms.chaotic,
            "worms_density": 1000,
            "worms_duration": 6,
            "worms_kink": random.randint(125, 375),
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "generator": lambda settings: {
            "hue_range": .25 + random.random() * .75,
            "hue_rotation": random.random(),
            "octaves": random.randint(1, 3),
            "ridges": coin_flip(),
        },
    },

    "fibers": {
        "post": lambda settings: [Effect("fibers")]
    },

    "figments": {
        "layers": ["multires-low", "voronoi", "funhouse", "wormhole", "bloom", "contrast", "lens"],
        "settings": lambda: {
            "speed": .025,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_refract": .333 + random.random() * .333,
            "wormhole_stride": .02 + random.random() * .01,
            "wormhole_kink": 4,
        },
        "generator": lambda settings: {
            "freq": 2,
            "hue_range": 2,
            "lattice_drift": 1,
        }
    },

    "filthy": {
        "layers": ["grime", "scratches", "stray-hair"],
    },

    "financial-district": {
        "layers": ["voronoi", "bloom", "contrast", "desaturate"],
        "settings": lambda: {
            "dist_metric": distance.manhattan,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_point_distrib": point.random,
            "voronoi_nth": random.randint(1, 3),
            "voronoi_point_freq": 2,
        }
    },

    "flux-capacitor": {
        "layers": ["symmetry", "voronoi", "worms", "contrast", "bloom", "brightness", "contrast", "vignette-dark"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "contrast": 2.5,
            "speed": .333,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_inverse": True,
            "voronoi_point_freq": 1,
            "voronoi_nth": 0,
            "worms_alpha": .25,
            "worms_behavior": worms.meandering,
            "worms_density": .125,
            "worms_drunkenness": .125,
            "worms_duration": 8,
            "worms_stride": 1,
            "worms_stride_deviation": 0,
        }
    },

    "fossil-hunt": {
        "layers": ["voronoi", "refract-octaves", "posterize-outline", "desaturate", "dither"],
        "settings": lambda: {
            "posterize_levels": random.randint(3, 5),
            "refract_range": random.randint(2, 4) * .5,
            "refract_y_from_offset": True,
            "voronoi_alpha": .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_freq": 10,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 5),
            "lattice_drift": 1.0,
        }
    },

    "fractal-forms": {
        "layers": ["fractal-seed"],
        "settings": lambda: {
            "worms_kink": random.randint(256, 512),
        }
    },

    "fractal-seed": {
        "layers": ["multires-low", "worms", "density-map", "random-hue", "bloom", "shadow",
                   "contrast", "desaturate", "aberration"],
        "settings": lambda: {
            "speed": .05,
            "palette_name": None,
            "worms_behavior": random_member([worms.chaotic, worms.random]),
            "worms_alpha": .9 + random.random() * .1,
            "worms_density": random.randint(750, 1250),
            "worms_duration": random.randint(2, 3),
            "worms_kink": 1.0,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "generator": lambda settings: {
            "freq": random.randint(2, 3),
            "hue_range": 1.0 + random.random() * 3.0,
            "ridges": coin_flip(),
        }
    },

    "fractal-smoke": {
        "layers": ["fractal-seed"],
        "settings": lambda: {
            "worms_behavior": worms.random,
            "worms_stride": random.randint(96, 192),
        }
    },

    "fractile": {
        "layers": ["symmetry", "voronoi", "reverb", "contrast", "palette", "random-hue",
                   "rotate", "lens"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "reverb_iterations": random.randint(2, 4),
            "reverb_octaves": random.randint(2, 4),
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(0, 2),
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": random.randint(2, 3),
        },
    },

    "fundamentals": {
        "layers": ["voronoi", "derivative-post", "density-map", "desaturate", "dither"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(3, 5),
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_refract": .125 + random.random() * .0625,
        },
        "generator": lambda settings: {
            "freq": random.randint(3, 5),
        }
    },

    "funhouse": {
        "settings": lambda: {
            "warp_freq": [random.randint(2, 3), random.randint(1, 3)],
            "warp_octaves": random.randint(1, 4),
            "warp_range": .25 + random.random() * .125,
            "warp_signed_range": False,
            "warp_spline_order": interp.bicubic
        },
        "post": lambda settings: [
            Effect("warp",
                   displacement=settings["warp_range"],
                   freq=settings["warp_freq"],
                   octaves=settings["warp_octaves"],
                   signed_range=settings["warp_signed_range"],
                   spline_order=settings["warp_spline_order"])
        ]
    },

    "funky-glyphs": {
        "layers": ["value-mask", "refract-post", "contrast", "desaturate", "rotate", "lens", "dither"],
        "settings": lambda: {
            "mask": random_member([
                mask.alphanum_binary, mask.alphanum_numeric, mask.alphanum_hex,
                mask.lcd, mask.lcd_binary,
                mask.fat_lcd, mask.fat_lcd_binary, mask.fat_lcd_numeric, mask.fat_lcd_hex
            ]),
            "mask_repeat": random.randint(1, 6),
            "refract_range": .125 + random.random() * .125,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
        },
        "generator": lambda settings: {
            "distrib": random_member([distrib.ones, distrib.uniform]),
            "octaves": random.randint(1, 2),
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        }
    },

    "galalaga": {
        "layers": ["value-mask"],
        "settings": lambda: {
            "mask": mask.invaders_square,
            "mask_repeat": 4,
        },
        "generator": lambda settings: {
            "distrib": distrib.uniform,
            "hue_range": random.random() * 2.5,
            "spline_order": interp.constant,
        },
        "post": lambda settings: [
            Effect("glyph_map",
                   colorize=True,
                   mask=mask.invaders_square,
                   zoom=32.0),
            Effect("glyph_map",
                   colorize=True,
                   mask=random_member([mask.invaders_square, mask.rgb]),
                   zoom=4.0),
            Effect("normalize"),
            Preset("glitchin-out"),
            Preset("contrast"),
            Preset("crt"),
            Preset("lens"),
        ],
    },

    "game-show": {
        "layers": ["maybe-palette", "posterize", "be-kind-rewind"],
        "settings": lambda: {
            "posterize_levels": random.randint(2, 5),
        },
        "generator": lambda settings: {
            "distrib": distrib.normal,
            "freq": random.randint(8, 16) * 2,
            "mask": random_member([mask.h_tri, mask.v_tri]),
            "spline_order": interp.cosine,
        }
    },

    "glass-darkly": {
        "layers": ["multires-alpha", "nudge-hue", "reflect-post"],
        "settings": lambda: {
            "reflect_range": .95 + random.random() * .1,
        },
        "generator": lambda settings: {
            "distrib": distrib.lognormal,
            "octaves": 8,
            "rgb": True,
        },
    },

    "glitchin-out": {
        "layers": ["corrupt"],
        "post": lambda settings: [Effect("glitch"), Preset("crt"), Preset("bloom")]
    },

    "globules": {
        "layers": ["multires-low", "reflect-octaves", "density-map", "shadow", "lens", "contrast"],
        "settings": lambda: {
            "palette_name": None,
            "reflect_range": 2.5,
            "speed": .125,
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": random.randint(3, 6),
            "hue_range": .25 + random.random() * .5,
            "lattice_drift": 1,
            "mask": mask.sparse,
            "mask_static": True,
            "octaves": random.randint(3, 6),
            "saturation": .175 + random.random() * .175,
        }
    },

    "glom": {
        "layers": ["refract-octaves", "reflect-octaves", "refract-post", "reflect-post", "funhouse",
                   "bloom", "shadow", "bloom", "contrast", "lens", "contrast"],
        "settings": lambda: {
            "reflect_range": .125 + random.random() * .125,
            "refract_range": .125 + random.random() * .125,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "speed": .025,
            "warp_range": .125 + random.random() * .125,
            "warp_octaves": 1,
        },
        "generator": lambda settings: {
            "freq": 2,
            "hue_range": .25 + random.random() * .25,
            "lattice_drift": 1,
            "octaves": 2,
        }
    },

    "glowing-edges": {
        "post": lambda settings: [Effect("glowing_edges")]
    },

    "glyph-map": {
        "settings": lambda: {
            "glyph_map_colorize": coin_flip(),
            "glyph_map_mask": random_member(set(mask.procedural_members()).intersection(masks.square_masks())),
            "glyph_map_zoom": random.randint(1, 3),
        },
        "post": lambda settings: [
            Effect("glyph_map",
                   colorize=settings["glyph_map_colorize"],
                   mask=settings["glyph_map_mask"],
                   zoom=settings["glyph_map_zoom"])
        ]
    },

    "glyphic": {
        "layers": ["value-mask", "posterize", "palette", "desaturate", "maybe-invert", "dither", "distressed"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
            "posterize_levels": 1,
        },
        "generator": lambda settings: {
            "corners": True,
            "mask": settings["mask"],
            "freq": masks.mask_shape(settings["mask"])[0:2],
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(3, 5),
            "saturation": 0,
            "spline_order": interp.cosine,
        },
    },

    "graph-paper": {
        "layers": ["voronoi", "derivative-post", "rotate", "crt", "lens", "bloom", "contrast"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "voronoi_alpha": .5 + random.random() * .25,
            "voronoi_refract": 1.5 + random.random() * .75,
            "voronoi_refract_y_from_offset": True,
            "voronoi_diagram_type": voronoi.flow,
        },
        "generator": lambda settings: {
            "corners": True,
            "distrib": distrib.ones,
            "freq": random.randint(4, 8) * 2,
            "mask": mask.chess,
            "rgb": True,
            "spline_order": interp.constant,
        }
    },

    "grass": {
        "layers": ["multires", "worms", "dither", "lens", "contrast"],
        "settings": lambda: {
            "worms_behavior": random_member([worms.chaotic, worms.meandering]),
            "worms_alpha": .9,
            "worms_density": 50 + random.random() * 25,
            "worms_drunkenness": .125,
            "worms_duration": 1.125,
            "worms_stride": .875,
            "worms_stride_deviation": .125,
            "worms_kink": .125 + random.random() * .5,
        },
        "generator": lambda settings: {
            "freq": random.randint(6, 12),
            "hue_rotation": .25 + random.random() * .05,
            "lattice_drift": 1,
            "saturation": .625 + random.random() * .25,
        }
    },

    "grayscale": {
        "post": lambda settings: [Effect("adjust_saturation", amount=0)]
    },

    "grime": {
        "post": lambda settings: [Effect("grime")]
    },

    "groove-is-stored-in-the-heart": {
        "layers": ["posterize", "ripples", "distressed"],
        "settings": lambda: {
            "posterize_levels": random.randint(1, 2),
            "ripple_range": .75 + random.random() * .375,
        },
        "generator": lambda settings: {
            "distrib": distrib.column_index,
        },
    },

    "halt-catch-fire": {
        "layers": ["multires-low"],
        "generator": lambda settings: {
            "freq": 2,
            "hue_range": .05,
            "lattice_drift": 1,
            "spline_order": interp.constant,
        },
        "post": lambda settings: [
            Effect("glitch"),
            Preset("pixel-sort"),
            Preset("rotate"),
            Preset("crt"),
            Preset("vignette-dark"),
            Preset("contrast")
        ]
    },

    "heartburn": {
        "layers": ["voronoi", "bloom", "vignette-dark", "contrast"],
        "settings": lambda: {
            "contrast": 10 + random.random() * 5.0,
            "dist_metric": random_member(distance.all()),
            "voronoi_alpha": 0.9625,
            "voronoi_diagram_type": 42,
            "voronoi_inverse": True,
            "voronoi_point_freq": 1,
        },
        "generator": lambda settings: {
            "freq": random.randint(12, 18),
            "octaves": random.randint(1, 3),
            "ridges": True,
        },
    },

    "hotel-carpet": {
        "layers": ["basic", "ripples", "carpet", "dither"],
        "settings": lambda: {
            "ripple_kink": .5 + random.random() * .25,
            "ripple_range": .666 + random.random() * .333,
        },
        "generator": lambda settings: {
            "spline_order": interp.constant,
        },
    },

    "hsv-gradient": {
        "layers": ["basic", "rotate", "lens"],
        "settings": lambda: {
            "palette_name": None
        },
        "generator": lambda settings: {
            "hue_range": .5 + random.random() * 2.0,
            "lattice_drift": 1.0,
        }
    },

    "hydraulic-flow": {
        "layers": ["multires", "derivative-octaves", "refract-octaves", "erosion-worms", "density-map",
                   "maybe-invert", "shadow", "bloom", "rotate", "dither", "lens"],
        "settings": lambda: {
            "deriv_alpha": .25 + random.random() * .25,
            "erosion_worms_alpha": .125 + random.random() * .125,
            "erosion_worms_contraction": .75 + random.random() * .5,
            "erosion_worms_density": random.randint(5, 250),
            "erosion_worms_iterations": random.randint(50, 250),
            "palette_name": None,
            "refract_range": random.random(),
        },
        "generator": lambda settings: {
            "distrib": random_member([m for m in distrib if m not in (distrib.ones, distrib.mids)]),
            "hue_range": random.random(),
            "ridges": coin_flip(),
            "rgb": coin_flip(),
            "saturation": random.random(),
        }
    },

    "i-made-an-art": {
        "layers": ["basic", "outline", "desaturate", "distressed", "contrast"],
        "generator": lambda settings: {
            "spline_order": interp.constant,
            "lattice_drift": random.randint(5, 10),
            "hue_range": random.random() * 4,
        }
    },

    "inkling": {
        "layers": ["voronoi", "refract-post", "funhouse", "grayscale", "density-map", "contrast",
                   "fibers", "grime", "scratches"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "contrast": 2.5,
            "refract_range": .25 + random.random() * .125,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_refract": .25 + random.random() * .125,
            "warp_range": .125 + random.random() * .0625,
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": random.randint(2, 4),
            "lattice_drift": 1.0,
            "mask": mask.dropout,
            "mask_static": True,
        },
    },

    "invert": {
        "post": lambda settings: [Effect("convolve", kernel=mask.conv2d_invert)]
    },

    "is-this-anything": {
        "layers": ["soup"],
        "settings": lambda: {
            "refract_range": 2.5 + random.random() * 1.25,
            "voronoi_point_freq": 1,
        }
    },

    "kaleido": {
        "layers": ["wobble"],
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "kaleido_point_corners": False,
            "kaleido_point_distrib": point.random,
            "kaleido_point_freq": 1,
            "kaleido_sides": random.randint(5, 32),
            "kaleido_blend_edges": coin_flip(),
        },
        "post": lambda settings: [
            Effect("kaleido",
                   blend_edges=settings["kaleido_blend_edges"],
                   dist_metric=settings["dist_metric"],
                   point_corners=settings["kaleido_point_corners"],
                   point_distrib=settings["kaleido_point_distrib"],
                   point_freq=settings["kaleido_point_freq"],
                   sides=settings["kaleido_sides"]),
        ]
    },

    "lens": {
        "layers": ["normalize", "aberration", "vaseline", "tint", "vignette-dark"],
        "settings": lambda: {
            "vignette_alpha": .333 + random.random() * .16667
        }
    },

    "lens-warp": {
        "post": lambda settings: [Effect("lens_warp", displacement=.125 + random.random() * .125)]
    },

    "light-leak": {
        "layers": ["vignette-bright"],
        "post": lambda settings: [Effect("light_leak", alpha=.333 + random.random() * .333), Preset("bloom")]
    },

    "lowpoly": {
        "post": lambda settings: [Effect("lowpoly", freq=15)]
    },

    "mad-multiverse": {
        "layers": ["kaleido"],
        "settings": lambda: {
            "kaleido_point_freq": random.randint(3, 6),
        },
    },

    "maybe-derivative-post": {
        "post": lambda settings: [] if coin_flip() else [Preset("derivative-post")]
    },

    "maybe-invert": {
        "post": lambda settings: [] if coin_flip() else [Preset("invert")]
    },

    "maybe-palette": {
        "settings": lambda: {
            "palette_name": random_member(PALETTES)
        },
        "post": lambda settings: [] if coin_flip() else [Effect("palette", name=settings["palette_name"])]
    },

    "mosaic": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "voronoi_alpha": .75 + random.random() * .25
        },
        "post": lambda settings: [Preset("bloom")]
    },

    "multires": {
        "layers": ["basic"],
        "generator": lambda settings: {
            "octaves": random.randint(4, 8)
        }
    },

    "multires-alpha": {
        "layers": ["multires"],
        "settings": lambda: {
            "palette_name": None,
        },
        "generator": lambda settings: {
            "distrib": distrib.exp,
            "lattice_drift": 1,
            "octave_blending": blend.alpha,
            "octaves": 5,
        }
    },

    "multires-low": {
        "layers": ["basic"],
        "generator": lambda settings: {
            "octaves": random.randint(2, 4)
        }
    },

    "multires-ridged": {
        "layers": ["multires"],
        "generator": lambda settings: {
            "ridges": True
        }
    },

    "nebula": {
        "post": lambda settings: [Effect("nebula")]
    },

    "nerdvana": {
        "layers": ["symmetry", "voronoi", "density-map", "reverb", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "palette_name": None,
            "reverb_octaves": 2,
            "reverb_ridges": False,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": random.randint(5, 10),
            "voronoi_nth": 1,
        },
    },

    "noirmaker": {
        "layers": ["dither", "grayscale"],
        "post": lambda settings: [
            Effect("light_leak", alpha=.333 + random.random() * .333),
            Preset("bloom"),
            Preset("vignette-dark"),
            Effect("adjust_contrast", amount=5)
        ]
    },

    "normals": {
        "post": lambda settings: [Effect("normal_map")]
    },

    "normalize": {
        "post": lambda settings: [Effect("normalize")]
    },

    "nudge-hue": {
        "post": lambda settings: [Effect("adjust_hue", amount=-.125)]
    },

    "octave-warp": {
        "settings": lambda: {
            "speed": .025 + random.random() * .0125
        },
        "post": lambda settings: [Effect("warp", displacement=3.0 + random.random(), freq=random.randint(2, 3), octaves=3)]
    },

    "one-art-please": {
        "layers": ["dither", "light-leak", "contrast"],
        "post": lambda settings: [
            Effect("adjust_saturation", amount=.75),
            Effect("texture")
        ]
    },

    "outline": {
        "post": lambda settings: [Effect("outline", sobel_metric=distance.euclidean)]
    },

    "paintball-party": {
        "layers": ["spatter"] * random.randint(5, 7) + ["bloom"],
        "generator": lambda settings: {
            "distrib": distrib.zeros,
        }
    },

    "palette": {
        "settings": lambda: {
            "palette_name": random_member(PALETTES)
        },
        "post": lambda settings: [Effect("palette", name=settings["palette_name"])]
    },

    "pixel-sort": {
        "settings": lambda: {
            "pixel_sort_angled": coin_flip(),
            "pixel_sort_darkest": coin_flip(),
        },
        "post": lambda settings: [
            Effect("pixel_sort",
                   angled=settings["pixel_sort_angled"],
                   darkest=settings["pixel_sort_darkest"])
        ]
    },

    "polar": {
        "layers": ["kaleido"],
        "settings": lambda: {
            "kaleido_sides": 1
        },
    },

    "posterize": {
        "layers": ["normalize"],
        "settings": lambda: {
            "posterize_levels": random.randint(3, 7)
        },
        "post": lambda settings: [Effect("posterize", levels=settings["posterize_levels"])]
    },

    "posterize-outline": {
        "layers": ["posterize", "outline"]
    },

    "random-hue": {
        "post": lambda settings: [Effect("adjust_hue", amount=random.random())]
    },

    "reflect-octaves": {
        "settings": lambda: {
            "reflect_range": .5 + random.random() * 12.5,
        },
        "octaves": lambda settings: [
            Effect("refract",
                   displacement=settings["reflect_range"],
                   from_derivative=True)
        ]
    },

    "reflect-post": {
        "settings": lambda: {
            "reflect_range": .5 + random.random() * 12.5,
        },
        "post": lambda settings: [
            Effect("refract",
                   displacement=settings["reflect_range"],
                   from_derivative=True)
        ]
    },

    "refract-octaves": {
        "settings": lambda: {
            "refract_range": .5 + random.random() * .25,
            "refract_signed_range": True,
            "refract_y_from_offset": False,
        },
        "octaves": lambda settings: [
            Effect("refract",
                   displacement=settings["refract_range"],
                   signed_range=settings["refract_signed_range"],
                   y_from_offset=settings["refract_y_from_offset"])
        ]
    },

    "refract-post": {
        "settings": lambda: {
            "refract_range": .125 + random.random() * 1.25,
            "refract_signed_range": True,
            "refract_y_from_offset": True,
        },
        "post": lambda settings: [
            Effect("refract",
                   displacement=settings["refract_range"],
                   signed_range=settings["refract_signed_range"],
                   y_from_offset=settings["refract_y_from_offset"])
        ]
    },

    "reindex-octaves": {
        "settings": lambda: {
            "reindex_range": .125 + random.random() * 2.5
        },
        "octaves": lambda settings: [Effect("reindex", displacement=settings["reindex_range"])]
    },

    "reindex-post": {
        "settings": lambda: {
            "reindex_range": .125 + random.random() * 2.5
        },
        "post": lambda settings: [Effect("reindex", displacement=settings["reindex_range"])]
    },

    "reverb": {
        "layers": ["normalize"],
        "settings": lambda: {
            "reverb_iterations": 1,
            "reverb_ridges": coin_flip(),
            "reverb_octaves": random.randint(3, 6)
        },
        "post": lambda settings: [
            Effect("reverb",
                   iterations=settings["reverb_iterations"],
                   octaves=settings["reverb_octaves"],
                   ridges=settings["reverb_ridges"])
        ]
    },

    "ripples": {
        "settings": lambda: {
            "ripple_range": .025 + random.random() * .1,
            "ripple_freq": random.randint(2, 3),
            "ripple_kink": random.randint(3, 18)
        },
        "post": lambda settings: [
            Effect("ripple",
                   displacement=settings["ripple_range"],
                   freq=settings["ripple_freq"],
                   kink=settings["ripple_kink"])
        ]
    },

    "rotate": {
        "settings": lambda: {
            "angle": random.random() * 360.0
        },
        "post": lambda settings: [Effect("rotate", angle=settings["angle"])]
    },

    "scanline-error": {
        "post": lambda settings: [Effect("scanline_error")]
    },

    "scratches": {
        "post": lambda settings: [Effect("scratches")]
    },

    "scuff": {
        "post": lambda settings: [Effect("scratches")]
    },

    "shadow": {
        "settings": lambda: {
            "shadow_alpha": .5 + random.random() * .25
        },
        "post": lambda settings: [Effect("shadow", alpha=settings["shadow_alpha"])]
    },

    "shadows": {
        "layers": ["shadow", "vignette-dark"]
    },

    "shake-it-like": {
        "post": lambda settings: [Effect("frame")]
    },

    "simple-frame": {
        "post": lambda settings: [Effect("simple_frame")]
    },

    "sine-octaves": {
        "settings": lambda: {
            "sine_range": random.randint(4, 20)
        },
        "octaves": lambda settings: [
            Effect("sine", displacement=settings["sine_range"])
        ]
    },

    "sine-post": {
        "settings": lambda: {
            "sine_range": random.randint(4, 20)
        },
        "post": lambda settings: [
            Effect("sine", displacement=settings["sine_range"])
        ]
    },

    "sketch": {
        "post": lambda settings: [Effect("sketch"), Effect("fibers"), Effect("grime"), Effect("texture")]
    },

    "snow": {
        "post": lambda settings: [Effect("snow", alpha=.333 + random.random() * .16667)]
    },

    "sobel": {
        "post": lambda settings: [Effect("sobel", dist_metric=random_member(distance.all()))]
    },

    "soup": {
        "layers": ["voronoi", "normalize", "refract-post", "worms", "density-map", "bloom", "shadow", "lens", "contrast"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "refract_range": .5 + random.random() * .25,
            "refract_y_from_offset": True,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_inverse": True,
            "voronoi_point_freq": random.randint(2, 3),
            "worms_alpha": .75 + random.random() * .25,
            "worms_behavior": worms.random,
            "worms_density": 500,
            "worms_kink": 4.0 + random.random() * 2.0,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "generator": lambda settings: {
            "freq": random.randint(2, 3),
        }
    },

    "spatter": {
        "settings": lambda: {
            "speed": .0333 + random.random() * .016667
        },
        "post": lambda settings: [Effect("spatter")]
    },

    "spooky-ticker": {
        "post": lambda settings: [Effect("spooky_ticker")]
    },

    "starfield": {
        "layers": ["multires-low", "brightness", "nebula", "contrast", "lens", "dither", "vignette-dark"],
        "settings": lambda: {
            "brightness": -.075,
            "contrast": 4.0,
            "palette_name": None,
        },
        "generator": lambda settings: {
            "distrib": distrib.exp,
            "freq": random.randint(400, 500),
            "mask": mask.sparser,
            "mask_static": True,
            "spline_order": interp.linear,
        },
    },

    "stray-hair": {
        "post": lambda settings: [Effect("stray_hair")]
    },

    "subpixels": {
        "post": lambda settings: [Effect("glyph_map", mask=random_member(mask.rgb_members()), zoom=random_member([2, 4, 8]))]
    },

    "symmetry": {
        "layers": ["maybe-palette"],
        "generator": lambda settings: {
            "corners": True,
            "freq": [2, 2],
        },
    },

    "swerve-h": {
        "post": lambda settings: [
            Effect("warp",
                   displacement=.5 + random.random() * .5,
                   freq=[random.randint(2, 5), 1],
                   octaves=1,
                   spline_order=interp.bicubic)
        ]
    },

    "swerve-v": {
        "post": lambda settings: [
            Effect("warp",
                   displacement=.5 + random.random() * .5,
                   freq=[1, random.randint(2, 5)],
                   octaves=1,
                   spline_order=interp.bicubic)
        ]
    },

    "tensor-tone": {
        "post": lambda settings: [Effect("glyph_map", colorize=coin_flip())]
    },

    "tint": {
        "post": lambda settings: [Effect("tint", alpha=.25 + random.random() * .125)]
    },

    "value-mask": {
        "settings": lambda: {
            "mask": random_member(mask),
            "mask_repeat": random.randint(2, 8)
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": [int(i * settings["mask_repeat"]) for i in masks.mask_shape(settings["mask"])[0:2]],
            "mask": settings["mask"],
            "spline_order": random_member([m for m in interp if m != interp.bicubic])
        }
    },

    "vaseline": {
        "post": lambda settings: [Effect("vaseline", alpha=.625 + random.random() * .25)]
    },

    "vignette-bright": {
        "settings": lambda: {
            "vignette_alpha": .333 + random.random() * .333,
            "vignette_brightness": 1.0,
        },
        "post": lambda settings: [
            Effect("vignette",
                   alpha=settings["vignette_alpha"],
                   brightness=settings["vignette_brightness"])
        ]
    },

    "vignette-dark": {
        "settings": lambda: {
            "vignette_alpha": .65 + random.random() * .35,
            "vignette_brightness": 0.0,
        },
        "post": lambda settings: [
            Effect("vignette",
                   alpha=settings["vignette_alpha"],
                   brightness=settings["vignette_brightness"])
        ]
    },

    "voronoi": {
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": random_member([t for t in voronoi if t != voronoi.none]),
            "voronoi_inverse": False,
            "voronoi_nth": random.randint(0, 2),
            "voronoi_point_corners": False,
            "voronoi_point_distrib": point.random if coin_flip() else random_member(point, mask.nonprocedural_members()),
            "voronoi_point_drift": 0.0,
            "voronoi_point_freq": random.randint(4, 10),
            "voronoi_point_generations": 1,
            "voronoi_refract": 0,
            "voronoi_refract_y_from_offset": True,
        },
        "post": lambda settings: [
            Effect("voronoi",
                   alpha=settings["voronoi_alpha"],
                   diagram_type=settings["voronoi_diagram_type"],
                   dist_metric=settings["dist_metric"],
                   inverse=settings["voronoi_inverse"],
                   nth=settings["voronoi_nth"],
                   point_corners=settings["voronoi_point_corners"],
                   point_distrib=settings["voronoi_point_distrib"],
                   point_drift=settings["voronoi_point_drift"],
                   point_freq=settings["voronoi_point_freq"],
                   point_generations=settings["voronoi_point_generations"],
                   with_refract=settings["voronoi_refract"],
                   refract_y_from_offset=settings["voronoi_refract_y_from_offset"])
        ]
    },

    "voronoid": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "voronoi_refract": .25 + random.random() * .25
        }
    },

    "vortex": {
        "post": lambda settings: [Effect("vortex", displacement=random.randint(16, 48))]
    },

    "wobble": {
        "post": lambda settings: [Effect("wobble")]
    },

    "wormhole": {
        "settings": lambda: {
            "wormhole_kink": .5 + random.random(),
            "wormhole_stride": .025 + random.random() * .05
        },
        "post": lambda settings: [
            Effect("wormhole",
                   kink=settings["wormhole_kink"],
                   input_stride=settings["wormhole_stride"])
        ]
    },

    "worms": {
        "settings": lambda: {
            "worms_alpha": .75 + random.random() * .25,
            "worms_behavior": random_member(worms.all()),
            "worms_density": random.randint(250, 500),
            "worms_drunkenness": 0.0,
            "worms_drunken_spin": False,
            "worms_duration": .5 + random.random(),
            "worms_kink": 1.0 + random.random() * 1.5,
            "worms_stride": .75 + random.random() * .5,
            "worms_stride_deviation": random.random() + .5
        },
        "post": lambda settings: [
            Effect("worms",
                   alpha=settings["worms_alpha"],
                   behavior=settings["worms_behavior"],
                   density=settings["worms_density"],
                   drunkenness=settings["worms_drunkenness"],
                   drunken_spin=settings["worms_drunken_spin"],
                   duration=settings["worms_duration"],
                   kink=settings["worms_kink"],
                   stride=settings["worms_stride"],
                   stride_deviation=settings["worms_stride_deviation"])
        ]
    },

}

Preset = functools.partial(Preset, presets=PRESETS)
