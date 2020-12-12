import functools
import random

from noisemaker.composer import Effect, Preset
from noisemaker.constants import (
    DistanceMetric as distance,
    InterpolationType as interp,
    OctaveBlending as blend,
    PointDistribution as point,
    ValueDistribution as distrib,
    ValueMask,
    VoronoiDiagramType as voronoi,
    WormBehavior as worms,
)
from noisemaker.palettes import PALETTES
from noisemaker.presets import coin_flip, random_member

import noisemaker.masks as masks

#: A dictionary of presets for use with the artmaker-new script.
PRESETS = {
    "1969": {
        "extends": ["symmetry", "voronoi"],
        "settings": lambda: {
            "palette_name": None,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_dist_metric": distance.euclidean,
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.circular,
            "voronoi_point_freq": random.randint(3, 5) * 2,
            "voronoi_nth": random.randint(1, 3),
        },
        "generator": lambda settings: {
            "rgb": True,
        },
        "post": lambda settings: [
            Effect("normalize"),
            Preset("posterize-outline"),
            Preset("distressed")
        ]
    },

    "1976": {
        "extends": ["voronoi"],
        "settings": lambda: {
            "voronoi_point_freq": 2,
            "voronoi_dist_metric": distance.triangular,
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_nth": 0
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
            "voronoi_dist_metric": distance.chebyshev,
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
        "extends": ["value-mask"],
        "settings": lambda: {
            "vignette_alpha": .75 + random.random() * .25
        },
        "generator": lambda settings: {
            "freq": 13 * random.randint(10, 20),
            "mask": ValueMask.bank_ocr
        },
        "post": lambda settings: [
            Preset("invert"),
            Effect("posterize", levels=random.randint(1, 2)),
            Preset("vignette-bright", settings=settings),
            Preset("aberration"),
        ]
    },

    "2d-chess": {
        "extends": ["value-mask", "voronoi"],
        "settings": lambda: {
            "voronoi_alpha": 0.5 + random.random() * .5,
            "voronoi_diagram_type": voronoi.color_range if coin_flip() \
                else random_member([m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),  # noqa E131
            "voronoi_nth": random.randint(0, 1) * random.randint(0, 63),
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.square,
            "voronoi_dist_metric": random_member(distance.absolute_members()),
            "voronoi_point_freq": 8,
        },
        "generator": lambda settings: {
            "corners": True,
            "freq": 8,
            "mask": ValueMask.chess,
            "spline_order": interp.constant,
        }
    },

    "aberration": {
        "post": lambda settings: [Effect("aberration", displacement=.025 + random.random() * .0125)]
    },

    "basic": {
        "extends": ["maybe-palette"],
        "generator": lambda settings: {
            "freq": random.randint(2, 4),
        }
    },

    "be-kind-rewind": {
        "post": lambda settings: [Effect("vhs"), Preset("crt")]
    },

    "bloom": {
        "post": lambda settings: [Effect("bloom", alpha=.25 + random.random() * .125)]
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
        "post": lambda settings: [Effect("density_map"), Effect("convolve", kernel=ValueMask.conv2d_invert), Preset("dither")]
    },

    "derivative-octaves": {
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members())
        },
        "octaves": lambda settings: [Effect("derivative", dist_metric=settings["dist_metric"])]
    },

    "derivative-post": {
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members())
        },
        "post": lambda settings: [Effect("derivative", dist_metric=settings["dist_metric"])]
    },

    "desaturate": {
        "post": lambda settings: [Effect("adjust_saturation", amount=.333 + random.random() * .16667)]
    },

    "distressed": {
        "extends": ["dither", "filthy"],
        "post": lambda settings: [Preset("desaturate")]
    },

    "dither": {
        "post": lambda settings: [Effect("dither", alpha=.125 + random.random() * .06125)]
    },

    "erosion-worms": {
        "post": lambda settings: [
            Effect("erosion_worms",
                   alpha=.5 + random.random() * .5,
                   contraction=.5 + random.random() * .5,
                   density=random.randint(25, 100),
                   iterations=random.randint(25, 100)),
        ]
    },

    "falsetto": {
        "post": lambda settings: [Effect("false_color")]
    },

    "filthy": {
        "post": lambda settings: [Effect("grime"), Effect("scratches"), Effect("stray_hair")]
    },

    "funhouse": {
        "post": lambda settings: [
            Effect("warp",
                   displacement=.25 + random.random() * .125,
                   freq=[random.randint(2, 3), random.randint(1, 3)],
                   octaves=random.randint(1, 4),
                   signed_range=False,
                   spline_order=interp.bicubic)
        ]
    },

    "glitchin-out": {
        "extends": ["corrupt", "crt"],
        "post": lambda settings: [Effect("glitch"), Preset("bloom")]
    },

    "glowing-edges": {
        "post": lambda settings: [Effect("glowing_edges")]
    },

    "glyph-map": {
        "post": lambda settings: [Effect("glyph_map", colorize=coin_flip, zoom=random.randint(1, 3))]
    },

    "grayscale": {
        "post": lambda settings: [Effect("adjust_saturation", amount=0)]
    },

    "invert": {
        "post": lambda settings: [Effect("convolve", kernel=ValueMask.conv2d_invert)]
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
        "post": lambda settings: [Effect("lowpoly")]
    },

    "mad-multiverse": {
        "extends": ["kaleido"],
        "settings": lambda: {
            "point_freq": random.randint(3, 6),
        },
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
        },
        "post": lambda settings: [
            Effect("normalize")
        ]
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
        "extends": ["symmetry", "voronoi"],
        "settings": lambda: {
            "palette_name": None,
            "reverb_octaves": 2,
            "reverb_ridges": False,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_dist_metric": distance.euclidean,
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": random.randint(5, 10),
            "voronoi_nth": 1,
        },
        "post": lambda settings: [
            Effect("normalize"),
            Preset("density-map"),
            Preset("reverb", settings=settings),
            Preset("bloom"),   # XXX Need a way for a final final pass. Bloom almost always comes last
        ]
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
        "extends": ["dither", "light-leak"],
        "post": lambda settings: [
            Effect("adjust_contrast", amount=1.25),
            Effect("adjust_saturation", amount=.75),
            Effect("texture")
        ]
    },

    "outline": {
        "post": lambda settings: [Effect("outline", sobel_metric=distance.euclidean)]
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

    "posterize-outline": {
        "post": lambda settings: [Effect("posterize", levels=random.randint(3, 7)), Preset("outline")]
    },

    "random-hue": {
        "post": lambda settings: [Effect("adjust_hue", amount=random.random())]
    },

    "reflect-domain-warp": {
        "post": lambda settings: [Effect("refract", displacement=.5 + random.random() * 12.5, from_derivative=True)]
    },

    "refract-domain-warp": {
        "post": lambda settings: [Effect("refract", displacement=.125 + random.random() * 1.25)]
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
        "post": lambda settings: [
            Effect("ripple",
                   displacement=.025 + random.random() * .1,
                   freq=random.randint(2, 3),
                   kink=random.randint(3, 18))
        ]
    },

    "rotate": {
        "post": lambda settings: [Effect("rotate", angle=random.random() * 360.0)]
    },

    "scanline-error": {
        "post": lambda settings: [Effect("scanline_error")]
    },

    "scuff": {
        "post": lambda settings: [Effect("scratches")]
    },

    "shadow": {
        "post": lambda settings: [Effect("shadow", alpha=.5 + random.random() * .25)]
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
        "extends": ["maybe-invert"],
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
        "post": lambda settings: [Effect("glyph_map", mask=random_member(ValueMask.rgb_members()), zoom=random_member([2, 4, 8]))]
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
            "value-mask-mask": random_member(ValueMask),
            "value-mask-repeat": random.randint(2, 8)
        },
        "generator": lambda settings: {
            "distrib": distrib.ones,
            "freq": [int(i * settings["value-mask-repeat"]) for i in masks.mask_shape(settings["value-mask-mask"])[0:2]],
            "mask": settings["value-mask-mask"],
            "spline_order": random_member([m for m in interp if m != interp.bicubic])
        }
    },

    "vaseline": {
        "post": lambda settings: [Effect("vaseline", alpha=.625 + random.random() * .25)]
    },

    "vignette-bright": {
        "settings": lambda: {
            "vignette_alpha": .333 + random.random() * .333
        },
        "post": lambda settings: [Effect("vignette", alpha=settings["vignette_alpha"], brightness=1)]
    },

    "vignette-dark": {
        "post": lambda settings: [Effect("vignette", alpha=.65 + random.random() * .35, brightness=0)]
    },

    "voronoi": {
        "settings": lambda: {
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": random_member([t for t in voronoi if t != voronoi.none]),
            "voronoi_dist_metric": random_member(distance.all()),
            "voronoi_inverse": coin_flip(),
            "voronoi_nth": random.randint(0, 2),
            "voronoi_point_corners": False,
            "voronoi_point_distrib": point.random if coin_flip() else random_member(point, ValueMask.nonprocedural_members()),
            "voronoi_point_drift": 0.0,
            "voronoi_point_freq": random.randint(4, 10),
            "voronoi_refract": 0
        },
        "post": lambda settings: [
            Effect("voronoi",
                   alpha=settings["voronoi_alpha"],
                   diagram_type=settings["voronoi_diagram_type"],
                   dist_metric=settings["voronoi_dist_metric"],
                   inverse=settings["voronoi_inverse"],
                   nth=settings["voronoi_nth"],
                   point_corners=settings["voronoi_point_corners"],
                   point_distrib=settings["voronoi_point_distrib"],
                   point_drift=settings["voronoi_point_drift"],
                   point_freq=settings["voronoi_point_freq"],
                   with_refract=settings["voronoi_refract"])
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
        "post": lambda settings: [Effect("wormhole", kink=.5 + random.random(), input_stride=.025 + random.random() * .05)]
    },

    "worms": {
        "settings": lambda: {
            "worms_alpha": .75 + random.random() * .25,
            "worms_behavior": random_member(worms.all()),
            "worms_density": random.randint(250, 500),
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
                   duration=settings["worms_duration"],
                   kink=settings["worms_kink"],
                   stride=settings["worms_stride"],
                   stride_deviation=settings["worms_stride_deviation"])
        ]
    },

}

Preset = functools.partial(Preset, presets=PRESETS)
