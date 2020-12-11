import functools
import random

from noisemaker.composer import Effect, Preset
from noisemaker.constants import (
    DistanceMetric as distance,
    InterpolationType as interp,
    ValueMask as mask,
    VoronoiDiagramType as voronoi,
    WormBehavior as worms,
)
from noisemaker.palettes import PALETTES
from noisemaker.presets import coin_flip, random_member

PRESETS = {
    "aberration": {
        "post": lambda settings: [Effect("aberration", displacement=.025 + random.random() * .0125)]
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
        "post": lambda settings: [Effect("density_map"), Effect("convolve", kernel=mask.conv2d_invert), Preset("dither")]
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
        "extends": ["bloom", "corrupt", "crt"],
        "post": lambda settings: [Effect("glitch")]
    },

    "glowing-edges": {
        "post": lambda settings: [Effect("glowing_edges")]
    },

    "glyph-map": {
        "post": lambda settings: [Effect("glyph_map", colorize=coin_flip, zoom=random.randint(1, 3))]
    },

    "grayscale": {
        "post": lambda settings: [Effect("adjust_saturation", 0)]
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
        "extends": ["bloom", "vignette-bright"],
        "post": lambda settings: [Effect("light_leak", alpha=.333 + random.random() * .333)]
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

    "maybe-palette": lambda: {
        "post": lambda settings: [] if coin_flip() else [Preset("palette")]
    },

    "mosaic": {
        "extends": ["bloom", "voronoi"],
        "post": lambda settings: [Effect("voronoi", alpha=.75 + random.random() * .25)]
    },

    "nebula": {
        "post": lambda settings: [Effect("nebula")]
    },

    "noirmaker": {
        "extends": ["bloom", "dither", "grayscale", "light-leak", "vignette-dark"],
        "post": lambda settings: [Effect("adjust_contrast", amount=5)]
    },

    "normals": {
        "post": lambda settings: [Effect("normal_map")]
    },

    "octave-warp": {
        "settings": {
            "speed": .0333,
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
        "post": lambda settings: [Effect("outline", dist_metric=distance.euclidean)]
    },

    "palette": {
        "post": lambda settings: [Effect("palette", name=random_member(PALETTES))]
    },

    "pixel-sort": {
        "post": lambda settings: [Effect("pixel_sort", angled=coin_flip(), darkest=coin_flip())]
    },

    "polar": {
        "extends": ["kaleido"],
        "settings": {
            "sides": 1
        },
    },

    "posterize-outline": {
        "extends": ["outline"],
        "post": lambda settings: Effect("posterize", levels=random.randint(3, 7))
    },

    "random-hue": {
        "post": lambda settings: [Effect("adjust_rotation", amount=random.random())]
    },

    "reflect-domain-warp": {
        "post": lambda settings: [Effect("refract", displacement=.5 + random.random() * 12.5, from_derivative=True)]
    },

    "refract-domain-warp": {
        "post": lambda settings: [Effect("refract", displacement=.125 + random.random() * 1.25)]
    },

    "reindex": {
        "post": lambda settings: [Effect("reindex", displacement=.125 + random.random() * 2.5)]
    },

    "reverb": {
        "post": lambda settings: [Effect("reverb", iterations=random.randint(1, 4), octaves=random.randint(3, 6))]
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
        "settings": {
            "speed": .05,
        },
        "post": lambda settings: [Effect("spatter")]
    },

    "spooky-ticker": {
        "post": lambda settings: [Effect("spooky_ticker")]
    },

    "subpixels": {
        "post": lambda settings: [Effect("composite", zoom=random_member([2, 4, 8]))]
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

    "vaseline": {
        "post": lambda settings: [Effect("vaseline", alpha=.625 + random.random() * .25)]
    },

    "vignette-bright": {
        "post": lambda settings: [Effect("vignette", alpha=.333 + random.random() * .333, brightness=1)]
    },

    "vignette-dark": {
        "post": lambda settings: [Effect("vignette", alpha=.65 + random.random() * .35, brightness=0)]
    },

    "voronoi": {
        "settings": lambda: {
            "voronoi_diagram_type": random_member([t for t in voronoi if t != voronoi.none]),
            "voronoi_point_freq": random.randint(4, 10),
            "voronoi_point_distrib": point.random if coin_flip() else random_member(point, mask.nonprocedural_members()),
            "voronoi_refract": 0
        },
        "post": lambda settings: [
            Effect("voronoi", 
                   diagram_type=settings["voronoi_diagram_type"],
                   dist_metric=random_member(distance.all()),
                   inverse=coin_flip(),
                   nth=random.randint(0, 2),
                   point_distrib=settings["voronoi_point_distrib"],
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
        "post": lambda settings: [Effect("wormhole", kink=.5 + random.random(), stride=.025 + random.random() * .05)]
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
