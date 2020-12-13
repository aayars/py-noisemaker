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
from noisemaker.presets import coin_flip, random_member

import noisemaker.masks as masks

#: A dictionary of presets for use with the artmaker-new script.
PRESETS = {
    "1969": {
        "extends": ["symmetry", "voronoi", "posterize-outline", "distressed"],
        "settings": lambda: {
            "palette_name": None,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_metric": distance.euclidean,
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
        "extends": ["voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_metric": distance.triangular,
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
        "extends": ["reindex", "voronoi"],
        "settings": lambda: {
            "reindex_range": .2 + random.random() * .1,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_metric": distance.chebyshev,
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
        "extends": ["analog-glitch", "invert", "posterize", "vignette-bright", "aberration"],
        "settings": lambda: {
            "mask": mask.bank_ocr,
            "mask_repeat": random.randint(9, 12),
            "vignette_alpha": .75 + random.random() * .25,
            "posterize_levels": random.randint(1, 2),
            "spline_order": interp.cosine,
        },
    },

    "2d-chess": {
        "extends": ["value-mask", "voronoi"],
        "settings": lambda: {
            "voronoi_alpha": 0.5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range if coin_flip() \
                else random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),  # noqa E131
            "voronoi_metric": random_member(distance.absolute_members()),
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
        "post": lambda settings: [Effect("aberration", displacement=.025 + random.random() * .0125)]
    },

    "abyssal-echoes": {
        "extends": ["multires-alpha", "desaturate", "random-hue"],
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
        "extends": ["reindex"],
        "generator": lambda settings: {
            "freq": random.randint(10, 15),
            "octaves": 8,
            "reindex_range": 1.25 + random.random() * 1.25,
            "rgb": True,
        },
        "post": lambda settings: [
            Effect("normalize")
        ]
    },

    "acid-droplets": {
        "extends": ["multires", "reflect-octaves", "density-map", "random-hue", "bloom", "shadow", "desaturate"],
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
        "extends": ["voronoid", "sobel", "funhouse", "bloom"],
        "settings": lambda: {
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_metric": distance.euclidean,
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": 4,
            "voronoi_point_generations": 2,
        },
        "generator": lambda settings: {
            "lattice_drift": coin_flip(),
        },
    },

    "acid-wash": {
        "extends": ["funhouse"],
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
        "extends": ["value-mask", "maybe-palette", "glitchin-out"],
        "generator": lambda settings: {
            "freq": 4,
            "mask": mask.white_bear,
            "rgb": coin_flip(),
            "spline_order": interp.constant,
        }
    },

    "aesthetic": {
        "extends": ["maybe-derivative-post", "spatter", "maybe-invert", "be-kind-rewind", "spatter"],
        "generator": lambda settings: {
            "corners": True,
            "distrib": random_member([distrib.column_index, distrib.ones, distrib.row_index]),
            "freq": random.randint(3, 5) * 2,
            "mask": mask.chess,
            "spline_order": interp.constant,
        },
    },

    "alien-terrain-multires": {
        "extends": ["multires-ridged", "maybe-invert", "bloom", "shadow", "desaturate"],
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
        "extends": ["multires-ridged", "invert", "voronoi", "derivative-octaves", "invert",
                    "erosion-worms", "bloom", "shadow", "dither", "boost-contrast", "desaturate"],
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
            "voronoi_metric": random_member(distance.absolute_members()),
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
        "extends": ["analog-glitch", "sobel", "glitchin-out"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
        }
    },

    "analog-glitch": {
        "extends": ["value-mask"],
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
        "extends": ["basic", "funhouse", "posterize"],
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
        },
        "post": lambda settings: [
            Effect("adjust_hue", amount=-.125),
            Preset("boost_contrast"),
            Preset("dither"),
        ],
    },

    "are-you-human": {
        "extends": ["multires", "value-mask", "funhouse", "density-map", "desaturate", "maybe-invert", "aberration", "snow"],
        "generator": lambda settings: {
            "freq": 15,
            "hue_range": random.random() * .25,
            "hue_rotation": random.random(),
            "mask": mask.truetype,
        },
    },

    "aztec-waffles": {
        "extends": ["symmetry", "voronoi", "maybe-invert", "reflect-post"],
        "settings": lambda: {
            "reflect_range": random.randint(12, 25),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_metric": random_member([distance.manhattan, distance.chebyshev]),
            "voronoi_nth": random.randint(2, 4),
            "voronoi_point_distrib": point.circular,
            "voronoi_point_freq": 4,
            "voronoi_point_generations": 2,
        },
    },

    "basic": {
        "extends": ["maybe-palette"],
        "generator": lambda settings: {
            "freq": random.randint(2, 4),
        },
        "post": lambda settings: [Effect("normalize")]
    },

    "basic-lowpoly": {
        "extends": ["basic", "lowpoly"],
    },

    "basic-voronoi": {
        "extends": ["basic", "voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": random_member([voronoi.color_range, voronoi.color_regions,
                                                   voronoi.range_regions, voronoi.color_flow])
        }
    },

    "basic-voronoi-refract": {
        "extends": ["basic", "voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": voronoi.range,
            "voronoi_metric": random_member(distance.absolute_members()),
            "voronoi_nth": 0,
            "voronoi_refract": 1.0 + random.random() * .5,
        },
        "generator": lambda settings: {
            "hue_range": .25 + random.random() * .5,
        },
    },

    "band-together": {
        "extends": ["reindex", "funhouse", "shadow"],
        "settings": lambda: {
            "reindex_range": random.randint(8, 12),
            "warp_range": .333 + random.random() * .16667,
            "warp_octaves": 8,
            "warp_freq": random.randint(2, 3),
        },
        "generator": lambda settings: {
            "freq": random.randint(6, 12),
        },
        "post": lambda settings: [Effect("normalize")]
    },

    "be-kind-rewind": {
        "post": lambda settings: [Effect("vhs"), Preset("crt")]
    },

    "beneath-the-surface": {
        "extends": ["multires-alpha", "reflect-octaves", "bloom", "shadow"],
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
        "extends": ["posterize", "maybe-palette", "funhouse", "distressed"],
        "settings": lambda: {
            "posterize_levels": 1,
            "warp_range": 1 + random.random() * .5,
        },
        "generator": lambda settings: {
            "distrib": distrib.column_index,
        },
    },

    "berkeley": {
        "extends": ["multires-ridged"],
        "settings": lambda: {
            "palette_name": None,
        },
        "generator": lambda settings: {
            "freq": random.randint(12, 16)
        },
        "octaves": lambda settings: [
            Effect("reindex", displacement=.75 + random.random() * .25),
            Effect("sine", displacement=2.0 + random.random() * 2.0),
        ],
        "post": lambda settings: [
            Effect("ridge"),
            Preset("shadow"),
            Preset("boost-contrast"),
        ]
    },

    "big-data-startup": {
        "extends": ["glyphic", "dither", "desaturate"],
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
        "extends": ["value-mask", "bloom", "crt"],
        "settings": lambda: {
            "mask": random_member([mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric]),
            "mask_repeat": random.randint(30, 60)
        }
    },

    "bitmask": {
        "extends": ["multires-low", "value-mask", "bloom"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
            "mask_repeat": random.randint(7, 15),
        },
        "generator": lambda settings: {
            "ridges": True,
        }
    },

    "blacklight-fantasy": {
        "extends": ["voronoi", "funhouse", "posterize", "sobel", "invert", "bloom", "dither"],
        "settings": lambda: {
            "posterize_levels": 3,
            "voronoi_metric": random_member(distance.absolute_members()),
            "voronoi_refract": .5 + random.random() * 1.25,
            "warp_octaves": random.randint(1, 4),
            "warp_range": random.randint(0, 1) * random.random(),
        },
        "generator": lambda settings: {
            "rgb": True,
        },
        "post": lambda settings: [Effect("adjust_hue", amount=-.125)]
    },

    "blockchain-stock-photo-background": {
        "extends": ["value-mask", "glitchin-out", "rotate", "vignette-dark"],
        "settings": lambda: {
            "angle": random.randint(5, 35),
            "vignette_alpha": 1.0,
            "mask": random_member([mask.alphanum_binary, mask.alphanum_hex,
                                   mask.alphanum_numeric, mask.bank_ocr]),
            "mask_repeat": random.randint(20, 40),
        },
    },

    "bloom": {
        "post": lambda settings: [Effect("bloom", alpha=.25 + random.random() * .125)]
    },

    "blotto": {
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "rgb": coin_flip(),
        },
        "post": lambda settings: [Effect("spatter", color=False), Preset("maybe-palette")]
    },

    "boost-contrast": {
        "settings": lambda: {
            "contrast": 1.25 + random.random() * .25
        },
        "post": lambda settings: [Effect("adjust_contrast", amount=settings["contrast"])]
    },

    "branemelt": {
        "extends": ["multires", "reflect-post", "bloom", "shadow"],
        "settings": lambda: {
            "palette_name": None,
            "reflect_range": .0333 + random.random() * .016667,
            "shadow_alpha": .666 + random.random() * .333,
        },
        "generator": lambda settings: {
            "freq": random.randint(12, 24),
        },
        "octaves": lambda settings: [
            Effect("sine", displacement=random.randint(64, 96)),
        ],
        "post": lambda settings: [
            Effect("adjust_brightness", amount=.125),
            Effect("adjust_contrast", amount=1.5),
        ]
    },

    "branewaves": {
        "extends": ["value-mask", "ripples", "bloom"],
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

    "bringing-hexy-back": {
        "extends": ["voronoi", "funhouse", "maybe-invert", "bloom"],
        "settings": lambda: {
            "voronoi_alpha": .333 + random.random() * .333,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_metric": distance.euclidean,
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
        "extends": ["multires-low", "reindex", "posterize", "glowing-edges", "dither", "desaturate"],
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
        "extends": ["worms", "tint", "boost-contrast", "bloom", "snow"],
        "settings": lambda: {
            "contrast": 3 + random.random() * 1.5,
            "palette_name": None,
            "worms_alpha": .875,
            "worms_behavior": worms.chaotic,
            "worms_stride_deviation": 0,
            "worms_density": .25 + random.random() * .125,
            "worms_drunken_spin": True,
            "worms_drunkenness": .1 + random.random() * .05,
            "worms_stride_deviation": 5.0 + random.random() * 5.0,
        },
        "generator": lambda settings: {
            "freq": 4,
            "hue_range": 2,
            "distrib": distrib.exp,
        },
    },

    "bubble-machine": {
        "extends": ["posterize", "wormhole", "reverb", "outline", "maybe-invert"],
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
        "extends": ["voronoi", "refract-post", "density-map", "random-hue", "bloom", "shadow"],
        "settings": lambda: {
            "refract_range": .125 + random.random() * .05,
            "speed": .05,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_metric": distance.euclidean,
            "voronoi_point_freq": 10,
            "voronoi_refract": .625 + random.random() * .25,
        },
    },

    "carpet": {
        "extends": ["worms"],
        "settings": lambda: {
            "worms_alpha": .25 + random.random() * .25,
            "worms_behavior": worms.chaotic,
            "worms_stride": .333 + random.random() * .333,
            "worms_stride_deviation": .25
        },
        "post": lambda settings: [Effect("grime")]
    },

    "celebrate": {
        "extends": ["posterize", "maybe-palette", "distressed"],
        "settings": lambda: {
            "posterize_levels": random.randint(3, 5),
            "speed": .025,
        },
        "generator": lambda settings: {
            "brightness_distrib": distrib.ones,
            "hue_range": 1,
        }
    },

    "cobblestone": {
        "extends": ["bringing-hexy-back", "desaturate"],
        "settings": lambda: {
            "saturation": .1 + random.random() * .05,
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
            Preset("shadow", settings=settings),
            Effect("adjust_brightness", amount=-.125),
            Preset("boost-contrast"),
            Effect("bloom", alpha=1.0)
        ],
    },

    "clouds": {
        "post": lambda settings: [Effect("clouds"), Preset("bloom"), Preset("dither")]
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
        "post": lambda settings: [Effect("crt")]
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

    "distressed": {
        "extends": ["dither", "filthy"],
        "post": lambda settings: [Preset("desaturate")]
    },

    "dither": {
        "post": lambda settings: [Effect("dither", alpha=.125 + random.random() * .06125)]
    },

    "erosion-worms": {
        "settings": lambda: {
            "erosion_worms_alpha": .5 + random.random() * .5,
            "erosion_worms_contraction": .5 + random.random() * .5,
            "erosion_worms_density": random.randint(25, 100),
            "erosion_worms_iterations": random.randint(25, 100),
            "erosion_worms_xy_blend": .75 + random.random() * .25
        },
        "post": lambda settings: [
            Effect("erosion_worms",
                   alpha=settings["erosion_worms_alpha"],
                   contraction=settings["erosion_worms_contraction"],
                   density=settings["erosion_worms_density"],
                   iterations=settings["erosion_worms_iterations"],
                   xy_blend=settings["erosion_worms_xy_blend"])
        ]
    },

    "falsetto": {
        "post": lambda settings: [Effect("false_color")]
    },

    "filthy": {
        "post": lambda settings: [Effect("grime"), Effect("scratches"), Effect("stray_hair")]
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

    "glitchin-out": {
        "extends": ["corrupt"],
        "post": lambda settings: [Effect("glitch"), Preset("crt"), Preset("bloom")]
    },

    "glowing-edges": {
        "post": lambda settings: [Effect("glowing_edges")]
    },

    "glyph-map": {
        "post": lambda settings: [Effect("glyph_map", colorize=coin_flip, zoom=random.randint(1, 3))]
    },

    "glyphic": {
        "extends": ["value-mask", "posterize", "maybe-invert"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
            "posterize_levels": 1,
        },
        "generator": lambda settings: {
            "mask": settings["mask"],
            "freq": masks.mask_shape(settings["mask"])[0:2],
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(3, 5),
            "saturation": 0,
            "spline_order": interp.cosine,
        },
    },

    "grayscale": {
        "post": lambda settings: [Effect("adjust_saturation", amount=0)]
    },

    "invert": {
        "post": lambda settings: [Effect("convolve", kernel=mask.conv2d_invert)]
    },

    "kaleido": {
        "extends": ["wobble"],
        "settings": lambda: {
            "point_freq": 1,
            "sides": random.randint(5, 32)
        },
        "post": lambda settings: [
            Effect("kaleido",
                   blend_edges=coin_flip(),
                   dist_metric=random_member(distance.all()),
                   point_freq=settings["point_freq"],
                   sides=settings["sides"])
        ]
    },

    "lens": {
        "extends": ["aberration", "vaseline", "tint"],
        "post": lambda settings: [Effect("vignette", alpha=.125 + random.random() * .125)]
    },

    "lens-warp": {
        "post": lambda settings: [Effect("lens_warp", displacement=.125 + random.random() * .125)]
    },

    "light-leak": {
        "extends": ["vignette-bright"],
        "post": lambda settings: [Effect("light_leak", alpha=.333 + random.random() * .333), Preset("bloom")]
    },

    "lowpoly": {
        "post": lambda settings: [Effect("lowpoly", freq=15)]
    },

    "mad-multiverse": {
        "extends": ["kaleido"],
        "settings": lambda: {
            "point_freq": random.randint(3, 6),
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
        "extends": ["voronoi"],
        "settings": lambda: {
            "voronoi_alpha": .75 + random.random() * .25
        },
        "post": lambda settings: [Preset("bloom")]
    },

    "multires": {
        "extends": ["basic"],
        "generator": lambda settings: {
            "octaves": random.randint(4, 8)
        }
    },

    "multires-alpha": {
        "extends": ["multires"],
        "settings": lambda: {
            "palette_name": None
        },
        "generator": lambda settings: {
            "distrib": distrib.exp,
            "lattice_drift": 1,
            "octave_blending": blend.alpha,
            "octaves": 5,
        }
    },

    "multires-low": {
        "extends": ["basic"],
        "generator": lambda settings: {
            "octaves": random.randint(2, 4)
        }
    },

    "multires-ridged": {
        "extends": ["multires"],
        "generator": lambda settings: {
            "ridges": True
        }
    },

    "nebula": {
        "post": lambda settings: [Effect("nebula")]
    },

    "nerdvana": {
        "extends": ["symmetry", "voronoi", "density-map", "reverb", "bloom"],
        "settings": lambda: {
            "palette_name": None,
            "reverb_octaves": 2,
            "reverb_ridges": False,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_metric": distance.euclidean,
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": random.randint(5, 10),
            "voronoi_nth": 1,
        },
    },

    "noirmaker": {
        "extends": ["dither", "grayscale"],
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

    "octave-warp": {
        "settings": lambda: {
            "speed": .025 + random.random() * .0125
        },
        "post": lambda settings: [Effect("warp", displacement=3.0 + random.random(), freq=random.randint(2, 3), octaves=3)]
    },

    "one-art-please": {
        "extends": ["dither", "light-leak", "boost-contrast"],
        "post": lambda settings: [
            Effect("adjust_saturation", amount=.75),
            Effect("texture")
        ]
    },

    "outline": {
        "post": lambda settings: [Effect("outline", sobel_metric=distance.euclidean)]
    },

    "paintball-party": {
        "extends": ["spatter"] * random.randint(5, 7) + ["bloom"],
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
        "post": lambda settings: [Effect("pixel_sort", angled=coin_flip(), darkest=coin_flip())]
    },

    "polar": {
        "extends": ["kaleido"],
        "settings": lambda: {
            "sides": 1
        },
    },

    "posterize": {
        "settings": lambda: {
            "posterize_levels": random.randint(3, 7)
        },
        "post": lambda settings: [Effect("normalize"), Effect("posterize", levels=settings["posterize_levels"])]
    },

    "posterize-outline": {
        "extends": ["posterize", "outline"]
    },

    "random-hue": {
        "post": lambda settings: [Effect("adjust_hue", amount=random.random())]
    },

    "reflect-octaves": {
        "settings": lambda: {
            "reflect_range": .5 + random.random() * 12.5
        },
        "octaves": lambda settings: [Effect("refract", displacement=settings["reflect_range"], from_derivative=True)]
    },

    "reflect-post": {
        "settings": lambda: {
            "reflect_range": .5 + random.random() * 12.5
        },
        "post": lambda settings: [Effect("refract", displacement=settings["reflect_range"], from_derivative=True)]
    },

    "refract-octaves": {
        "settings": lambda: {
            "refract_range": .125 + random.random() * 1.25
        },
        "octaves": lambda settings: [Effect("refract", displacement=settings["reflect_range"])]
    },

    "refract-post": {
        "settings": lambda: {
            "refract_range": .125 + random.random() * 1.25
        },
        "post": lambda settings: [Effect("refract", displacement=settings["refract_range"])]
    },

    "reindex": {
        "settings": lambda: {
            "reindex_range": .125 + random.random() * 2.5
        },
        "octaves": lambda settings: [Effect("reindex", displacement=settings["reindex_range"])]
    },

    "reverb": {
        "settings": lambda: {
            "reverb_iterations": random.randint(1, 4),
            "reverb_ridges": True,
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
            "ripples_range": .025 + random.random() * .1,
            "ripples_freq": random.randint(2, 3),
            "ripples_kink": random.randint(3, 18)
        },
        "post": lambda settings: [
            Effect("ripple",
                   displacement=settings["ripples_range"],
                   freq=settings["ripples_freq"],
                   kink=settings["ripples_kink"])
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
        "extends": ["shadow", "vignette-dark"]
    },

    "shake-it-like": {
        "post": lambda settings: [Effect("frame")]
    },

    "simple-frame": {
        "post": lambda settings: [Effect("simple_frame")]
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

    "spatter": {
        "settings": lambda: {
            "speed": .0333 + random.random() * .016667
        },
        "post": lambda settings: [Effect("spatter")]
    },

    "spooky-ticker": {
        "post": lambda settings: [Effect("spooky_ticker")]
    },

    "subpixels": {
        "post": lambda settings: [Effect("glyph_map", mask=random_member(mask.rgb_members()), zoom=random_member([2, 4, 8]))]
    },

    "symmetry": {
        "extends": ["maybe-palette"],
        "generator": lambda settings: {
            "corners": True,
            "freq": 2,
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
        "post": lambda settings: [Effect("tint", alpha=.333 + random.random() * .333)]
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
        "extends": ["vignette-bright"],
        "settings": lambda: {
            "vignette_alpha": .65 + random.random() * .35,
            "vignette_brightness": 0.0,
        },
    },

    "voronoi": {
        "settings": lambda: {
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": random_member([t for t in voronoi if t != voronoi.none]),
            "voronoi_inverse": False,
            "voronoi_metric": random_member(distance.all()),
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
                   dist_metric=settings["voronoi_metric"],
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
        "extends": ["voronoi"],
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
