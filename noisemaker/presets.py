import functools
import random

from noisemaker.composer import Effect, Preset, coin_flip, enum_range, random_member, stash
from noisemaker.constants import (
    ColorSpace as color,
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

#: Composable presets for Noisemaker. See composer.py and https://noisemaker.readthedocs.io/en/latest/composer.html
PRESETS = lambda: {  # noqa E731
    "1969": {
        "layers": ["symmetry", "voronoi", "posterize-outline", "distressed"],
        "settings": lambda: {
            "color_space": color.rgb,
            "dist_metric": distance.euclidean,
            "palette_on": False,
            "voronoi_alpha": 0.5 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.circular,
            "voronoi_point_freq": random.randint(3, 5) * 2,
            "voronoi_nth": random.randint(1, 3),
        },
        "ai": {
            "prompt": "1960s, kaleidoscopic, psychedelia, vintage poster art",
            "image_strength": 0.625,
            "cfg_scale": 25,
        }
    },

    "1976": {
        "layers": ["voronoi", "grain", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "saturation_final": 0.25 + random.random() * 0.125,
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": 2,
        },
        "ai": {
            "prompt": "1970s, overlapping shapes, vintage poster art",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "1985": {
        "layers": ["reindex-post", "voronoi", "palette", "random-hue", "spatter-post", "be-kind-rewind",
                   "spatter-final"],
        "settings": lambda: {
            "dist_metric": distance.chebyshev,
            "freq": random.randint(10, 15),
            "reindex_range": 0.2 + random.random() * 0.1,
            "spline_order": interp.constant,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.random,
            "voronoi_refract": 0.2 + random.random() * 0.1
        },
        "ai": {
            "prompt": "1980s, nostalgia, colorful, shapes, loud, retro",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "2001": {
        "layers": ["analog-glitch", "invert", "posterize", "vignette-bright", "aberration"],
        "settings": lambda: {
            "mask": mask.bank_ocr,
            "mask_repeat": random.randint(9, 12),
            "spline_order": interp.cosine,
            "vignette_alpha": 0.75 + random.random() * 0.25,
            "posterize_levels": random.randint(1, 2),
        },
        "ai": {
            "prompt": "2000s, retro futuristic, ocr, glitchy",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "2d-chess": {
        "layers": ["value-mask", "voronoi", "maybe-rotate"],
        "settings": lambda: {
            "corners": True,
            "dist_metric": random_member(distance.absolute_members()),
            "freq": 8,
            "mask": mask.chess,
            "spline_order": interp.constant,
            "voronoi_alpha": 0.5 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.color_range if coin_flip() else random_member(
                    [m for m in voronoi if not voronoi.is_flow_member(m) and m != voronoi.none]),  # noqa E131
            "voronoi_nth": random.randint(0, 1) * random.randint(0, 63),
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.square,
            "voronoi_point_freq": 8,
        },
        "ai": {
            "prompt": "decorative tile design, chessboard",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "aberration": {
        "settings": lambda: {
            "aberration_displacement": 0.0125 + random.random() * 0.000625
        },
        "final": lambda settings: [
            Effect("aberration",
                displacement=settings["aberration_displacement"])]
    },

    "acid": {
        "layers": ["basic", "reindex-post", "normalize"],
        "settings": lambda: {
            "color_space": color.rgb,
            "freq": random.randint(10, 15),
            "octaves": 8,
            "reindex_range": 1.25 + random.random() * 1.25,
        },
        "ai": {
            "prompt": "old textured wall, exposed layers of old paint, paint chipping and peeling away",
            "image_strength": 0.625,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "acid-droplets": {
        "layers": ["multires", "reflect-octaves", "density-map", "random-hue", "bloom", "shadow", "saturation"],
        "settings": lambda: {
            "freq": random.randint(8, 12),
            "hue_range": 0,
            "lattice_drift": 1.0,
            "mask": mask.sparse,
            "mask_static": True,
            "palette_on": False,
            "reflect_range": 7.5 + random.random() * 3.5
        },
        "ai": {
            "prompt": "corrosive liquid splatter, faded",
            "image_strength": 0.625,
            "cfg_scale": 25,
        }
    },

    "acid-grid": {
        "layers": ["voronoi-refract", "sobel", "funhouse", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "lattice_drift": coin_flip(),
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": 4,
            "voronoi_point_generations": 2,
            "warp_range": 0.125 + random.random() * 0.0625,
        },
        "ai": {
            "prompt": "psychedelic imagery, distorted geometric grid",
            "cfg_scale": 30,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "acid-wash": {
        "layers": ["basic", "funhouse", "ridge", "shadow", "saturation"],
        "settings": lambda: {
            "freq": random.randint(4, 6),
            "hue_range": 1.0,
            "ridges": True,
            "warp_octaves": 8,
        },
        "ai": {
            "prompt": "acid washed fabric",
        }
    },

    "activation-signal": {
        "layers": ["value-mask", "glitchin-out"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "freq": 4,
            "mask": mask.white_bear,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "blocky shape generated by analog computer, low quality, glitchy",
        }
    },

    "aesthetic": {
        "layers": ["basic", "maybe-derivative-post", "spatter-post", "maybe-invert", "be-kind-rewind",
                   "spatter-final"],
        "settings": lambda: {
            "corners": True,
            "distrib": random_member([distrib.column_index, distrib.ones, distrib.row_index]),
            "freq": random.randint(3, 5) * 2,
            "mask": mask.chess,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "1980s, 1990s, vaporwave, cyber, analog",
            "image_strength": 0.625,
            "cfg_scale": 25,
        }
    },

    "alien-terrain": {
        "layers": ["multires-ridged", "invert", "voronoi", "derivative-octaves", "invert",
                   "erosion-worms", "bloom", "shadow", "grain", "saturation"],
        "settings": lambda: {
            "grain_contrast": 1.5,
            "deriv_alpha": 0.25 + random.random() * 0.125,
            "dist_metric": distance.euclidean,
            "erosion_worms_alpha": 0.05 + random.random() * 0.025,
            "erosion_worms_density": random.randint(150, 200),
            "erosion_worms_inverse": True,
            "erosion_worms_xy_blend": 0.333 + random.random() * 0.16667,
            "freq": random.randint(3, 5),
            "hue_rotation": 0.875,
            "hue_range": 0.25 + random.random() * 0.25,
            "palette_on": False,
            "voronoi_alpha": 0.5 + random.random() * 0.25,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": 10,
            "voronoi_point_distrib": point.random,
            "voronoi_refract": 0.25 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "satellite photography, aerial photography, high detail, eroded sci-fi terrain, high-relief, geomorphology, arid desert, orbiter imagery, cratered, lunar surface",
            "style_preset": "photographic",
        }
    },

    "alien-glyphs": {
        "layers": ["entities", "maybe-rotate", "bloom", "crt"],
        "settings": lambda: {
            "corners": True,
            "mask": random_member([mask.arecibo_num, mask.arecibo_bignum, mask.arecibo_nucleotide]),
            "mask_repeat": random.randint(6, 12),
            "refract_range": 0.025 + random.random() * 0.0125,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "spline_order": random_member([interp.linear, interp.cosine]),
        },
        "ai": {
            "prompt": "sci-fi font, alien language glyphs",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "alien-transmission": {
        "layers": ["analog-glitch", "sobel", "glitchin-out"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
        },
        "ai": {
            "prompt": "sci-fi font, alien language glyphs",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "analog-glitch": {
        "layers": ["value-mask"],
        "settings": lambda: {
            # offset by i * 0.5 for glitched texture lookup
            "mask": random_member([mask.alphanum_hex, mask.lcd, mask.fat_lcd]),
            "mask_repeat": random.randint(20, 30),
        },
        "generator": lambda settings: {
            "freq": [int(i * 0.5 + i * settings["mask_repeat"]) for i in masks.mask_shape(settings["mask"])[0:2]],
        },
        "ai": {
            "prompt": "glitched broken alphanumeric lcd",
            "style_preset": "photographic"
        }
    },

    "arcade-carpet": {
        "layers": ["multires-alpha", "funhouse", "posterize", "nudge-hue", "carpet", "bloom", "contrast-final"],
        "settings": lambda: {
            "color_space": color.rgb,
            "distrib": distrib.exp,
            "hue_range": 1,
            "mask": mask.sparser,
            "mask_static": True,
            "octaves": 2,
            "palette_on": False,
            "posterize_levels": 3,
            "warp_freq": random.randint(25, 25),
            "warp_range": 0.03 + random.random() * 0.015,
            "warp_octaves": 1,
        },
        "generator": lambda settings: {
            "freq": settings["warp_freq"],
        },
        "ai": {
            "prompt": "blacklight arcade carpet, sci-fi, day-glow, bright colorful fluorescent shapes on black background, planets, stars, nebulas, comets, rockets, ufos, spaceships, asteroids, meteors, 1980s, 1990s",
            "image_strength": 0.25,
            "cfg_scale": 30,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "are-you-human": {
        "layers": ["multires", "value-mask", "funhouse", "density-map", "saturation", "maybe-invert", "aberration",
                   "snow"],
        "settings": lambda: {
            "freq": 15,
            "hue_range": random.random() * 0.25,
            "hue_rotation": random.random(),
            "mask": mask.truetype,
        },
        "ai": {
            "prompt": "an extremely difficult captcha",
        }
    },

    "band-together": {
        "layers": ["basic", "reindex-post", "funhouse", "shadow", "normalize", "grain"],
        "settings": lambda: {
            "freq": random.randint(6, 12),
            "reindex_range": random.randint(8, 12),
            "warp_range": 0.333 + random.random() * 0.16667,
            "warp_octaves": 8,
            "warp_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "artistic design with long bands of streaking color",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "basic": {
        "unique": True,
        "layers": ["maybe-palette"],
        "settings": lambda: {
            "brightness_distrib": None,
            "color_space": random_member(color.color_members()),
            "corners": False,
            "distrib": distrib.uniform,
            "freq": [random.randint(2, 4), random.randint(2, 4)],
            "hue_distrib": None,
            "hue_range": random.random() * 0.25,
            "hue_rotation": random.random(),
            "lattice_drift": 0.0,
            "mask": None,
            "mask_inverse": False,
            "mask_static": False,
            "octave_blending": blend.falloff,
            "octaves": 1,
            "ridges": False,
            "saturation": 1.0,
            "saturation_distrib": None,
            "sin": 0.0,
            "speed": 1.0,
            "spline_order": interp.bicubic,
        },
        "generator": lambda settings: {
            "brightness_distrib": settings["brightness_distrib"],
            "color_space": settings["color_space"],
            "corners": settings["corners"],
            "distrib": settings["distrib"],
            "freq": settings["freq"],
            "hue_distrib": settings["hue_distrib"],
            "hue_range": settings["hue_range"],
            "hue_rotation": settings["hue_rotation"],
            "lattice_drift": settings["lattice_drift"],
            "mask": settings["mask"],
            "mask_inverse": settings["mask_inverse"],
            "mask_static": settings["mask_static"],
            "octave_blending": settings["octave_blending"],
            "octaves": settings["octaves"],
            "ridges": settings["ridges"],
            "saturation": settings["saturation"],
            "saturation_distrib": settings["saturation_distrib"],
            "sin": settings["sin"],
            "spline_order": settings["spline_order"],
        },
        "ai": {
            "prompt": "a soft blend of several colors",
        }
    },

    "basic-low-poly": {
        "layers": ["basic", "low-poly", "grain", "saturation"],
        "ai": {
            "prompt": "a basic low-poly design, low poly mesh, diffuse lighting",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "low-poly",
        }
    },

    "basic-voronoi": {
        "layers": ["basic", "voronoi"],
        "settings": lambda: {
            "voronoi_diagram_type": random_member([voronoi.color_range, voronoi.color_regions,
                                                   voronoi.range_regions, voronoi.color_flow])
        },
        "ai": {
            "prompt": "a basic voronoi diagram"
        }
    },

    "basic-voronoi-refract": {
        "layers": ["basic", "voronoi"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "hue_range": 0.25 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": 0,
            "voronoi_refract": 1.0 + random.random() * 0.5,
        },
        "ai": {
            "prompt": "a basic voronoi diagram with cells that refract like glass",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "basic-water": {
        "layers": ["multires", "refract-octaves", "reflect-octaves", "ripple"],
        "settings": lambda: {
            "color_space": color.hsv,
            "distrib": distrib.uniform,
            "freq": random.randint(7, 10),
            "hue_range": 0.05 + random.random() * 0.05,
            "hue_rotation": 0.5125 + random.random() * 0.025,
            "lattice_drift": 1.0,
            "octaves": 4,
            "palette_on": False,
            "reflect_range": 0.16667 + random.random() * 0.16667,
            "refract_range": 0.25 + random.random() * 0.125,
            "refract_y_from_offset": True,
            "ripple_range": 0.005 + random.random() * 0.0025,
            "ripple_kink": random.randint(2, 4),
            "ripple_freq": random.randint(2, 4),
        },
        "ai": {
            "prompt": "simple water texture, ripples, waves, serene, calm",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "be-kind-rewind": {
        "final": lambda settings: [Effect("vhs"), Preset("crt")]
    },

    "benny-lava": {
        "layers": ["basic", "posterize", "funhouse", "distressed"],
        "settings": lambda: {
            "distrib": distrib.column_index,
            "posterize_levels": 1,
            "warp_range": 1 + random.random() * 0.5,
        },
        "ai": {
            "prompt": "melting and flowing posterized blobs",
        }
    },

    "berkeley": {
        "layers": ["multires-ridged", "reindex-octaves", "sine-octaves", "ridge", "shadow", "grain", "saturation"],
        "settings": lambda: {
            "freq": random.randint(12, 16),
            "palette_on": False,
            "reindex_range": 0.75 + random.random() * 0.25,
            "sine_range": 2.0 + random.random() * 2.0,
        },
        "ai": {
            "prompt": "psychedelia, vivid colors",
            "image_strength": 0.375,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "big-data-startup": {
        "layers": ["glyphic", "grain", "saturation"],
        "settings": lambda: {
            "mask": mask.script,
            "hue_rotation": random.random(),
            "hue_range": 0.0625 + random.random() * 0.5,
            "posterize_levels": random.randint(2, 4),
        },
        "ai": {
            "prompt": "horizontal cutout, corporate big data vibes, conference motif, futuristic pattern"
        }
    },

    "bit-by-bit": {
        "layers": ["value-mask", "bloom", "crt"],
        "settings": lambda: {
            "mask": random_member([mask.alphanum_binary, mask.alphanum_hex, mask.alphanum_numeric]),
            "mask_repeat": random.randint(20, 40)
        },
        "ai": {
            "prompt": "grid of glitched alphanumeric characters in a computer font"
        }
    },

    "bitmask": {
        "layers": ["multires-low", "value-mask", "bloom"],
        "settings": lambda: {
            "mask": random_member(mask.procedural_members()),
            "mask_repeat": random.randint(7, 15),
            "ridges": True,
        },
        "ai": {
            "prompt": "high detail grid of random glyphs in multiple sizes",
        }
    },

    "blacklight-fantasy": {
        "layers": ["voronoi", "funhouse", "posterize", "sobel", "invert", "bloom", "grain", "nudge-hue", 
                   "contrast-final"],
        "settings": lambda: {
            "color_space": color.rgb,
            "dist_metric": random_member(distance.absolute_members()),
            "posterize_levels": 3,
            "voronoi_refract": 0.5 + random.random() * 1.25,
            "warp_octaves": random.randint(1, 4),
            "warp_range": random.randint(0, 1) * random.random(),
        },
        "ai": {
            "prompt": "high detail brightly colored fluorescent blacklight outlines of a high fantasy scene over a black background",
            "image_strength": 0.375,
            "cfg_scale": 25,
            "style_preset": "fantasy-art",
        }
    },

    "bloom": {
        "settings": lambda: {
            "bloom_alpha": 0.05 + random.random() * 0.025,
        },
        "final": lambda settings: [
            Effect("bloom", alpha=settings["bloom_alpha"])
        ]
    },

    "blotto": {
        "layers": ["basic", "random-hue", "spatter-post", "maybe-palette", "maybe-invert"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "distrib": distrib.ones,
            "spatter_post_color": False,
        },
        "ai": {
            "prompt": "paint spattered on a solid background, spatter, splatter, splash, splat",
        }
    },

    "branemelt": {
        "layers": ["multires", "sine-octaves", "reflect-octaves", "bloom", "shadow", "grain", "saturation"],
        "settings": lambda: {
            "color_space": color.oklab,
            "freq": random.randint(6, 12),
            "palette_on": False,
            "reflect_range": 0.025 + random.random() * 0.0125,
            "shadow_alpha": 0.666 + random.random() * 0.333,
            "sine_range": random.randint(48, 64),
        },
        "ai": {
            "prompt": "psychedelic fractal imagery with overlapping ripples",
            "image_strength": 0.375,
            "cfg_scale": 25,
        }
    },

    "branewaves": {
        "layers": ["value-mask", "ripple", "bloom"],
        "settings": lambda: {
            "mask": random_member(mask.grid_members()),
            "mask_repeat": random.randint(5, 10),
            "ridges": True,
            "ripple_freq": 2,
            "ripple_kink": 1.5 + random.random() * 2,
            "ripple_range": 0.15 + random.random() * 0.15,
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        },
        "ai": {
            "prompt": "a brightly colored trippy psychedelic design with overlapping waves",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "brightness-post": {
        "settings": lambda: {
            "brightness_post": 0.125 + random.random() * 0.0625
        },
        "post": lambda settings: [Effect("adjust_brightness", amount=settings["brightness_post"])]
    },

    "brightness-final": {
        "settings": lambda: {
            "brightness_final": 0.125 + random.random() * 0.0625
        },
        "final": lambda settings: [Effect("adjust_brightness", amount=settings["brightness_final"])]
    },

    "bringing-hexy-back": {
        "layers": ["voronoi", "funhouse", "maybe-invert", "bloom"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "dist_metric": distance.euclidean,
            "hue_range": 0.25 + random.random() * 0.75,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_nth": 0,
            "voronoi_point_distrib": point.v_hex if coin_flip() else point.h_hex,
            "voronoi_point_freq": random.randint(4, 7) * 2,
            "warp_range": 0.05 + random.random() * 0.25,
            "warp_octaves": random.randint(1, 4),
        },
        "generator": lambda settings: {
            "freq": settings["voronoi_point_freq"],
        },
        "ai": {
            "prompt": "high detail brightly colored distorted hexagonal grid",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "broken": {
        "layers": ["multires-low", "reindex-octaves", "posterize", "glowing-edges", "grain", "saturation"],
        "settings": lambda: {
            "color_space": color.rgb,
            "freq": random.randint(3, 4),
            "lattice_drift": 2,
            "posterize_levels": 3,
            "reindex_range": random.randint(3, 4),
            "speed": 0.025,
        },
        "ai": {
            "prompt": "high detail brightly colored layers of broken shapes with glowing edges",
            "cfg_scale": 25,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "bubble-machine": {
        "layers": ["basic", "posterize", "wormhole", "reverb", "outline", "maybe-invert"],
        "settings": lambda: {
            "corners": True,
            "distrib": distrib.uniform,
            "freq": random.randint(3, 6) * 2,
            "mask": random_member([mask.h_hex, mask.v_hex]),
            "posterize_levels": random.randint(8, 16),
            "reverb_iterations": random.randint(1, 3),
            "reverb_octaves": random.randint(3, 5),
            "spline_order": random_member([m for m in interp if m != interp.constant]),
            "wormhole_stride": 0.1 + random.random() * 0.05,
            "wormhole_kink": 0.5 + random.random() * 4,
        },
        "ai": {
            "prompt": "billions of tiny bubbles",
        }
    },

    "bubble-multiverse": {
        "layers": ["voronoi", "refract-post", "density-map", "random-hue", "bloom", "shadow"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "refract_range": 0.125 + random.random() * 0.05,
            "speed": 0.05,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": 10,
            "voronoi_refract": 0.625 + random.random() * 0.25,
        },
        "ai": {
            "prompt": "high detail regions of color, complex fractal",
            "image_strength": 0.5,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "carpet": {
        "layers": ["worms", "grime"],
        "settings": lambda: {
            "worms_alpha": 0.25 + random.random() * 0.25,
            "worms_behavior": worms.chaotic,
            "worms_stride": 0.333 + random.random() * 0.333,
            "worms_stride_deviation": 0.25
        },
    },

    "celebrate": {
        "layers": ["basic", "posterize", "distressed"],
        "settings": lambda: {
            "brightness_distrib": distrib.ones,
            "hue_range": 1,
            "posterize_levels": random.randint(3, 5),
            "speed": 0.025,
        },
        "ai": {
            "prompt": "abstract design, art, vintage psychedelia",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "cell-reflect": {
        "layers": ["voronoi", "reflect-post", "derivative-post", "density-map", "maybe-invert",
                   "bloom", "grain", "saturation"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "palette_name": None,
            "reflect_range": random.randint(2, 4) * 5,
            "saturation_final": 0.5 + random.random() * 0.25,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": coin_flip(),
            "voronoi_point_distrib": random_member([m for m in point if m not in point.grid_members()]),
            "voronoi_point_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "reflective voronoi cells, mirrored finish, liquid metal, high detail fractal reflection",
            "style_preset": "photographic",
        }
    },

    "cell-refract": {
        "layers": ["voronoi", "ridge"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "dist_metric": random_member(distance.absolute_members()),
            "ridges": True,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_point_freq": random.randint(3, 4),
            "voronoi_refract": random.randint(8, 12) * 0.5,
        },
        "ai": {
            "prompt": "refractive voronoi cells, liquid glass, high detail fractal refraction",
            "image_strength": 0.5,
            "cfg_scale": 30,
            "style_preset": "photographic",
        }
    },

    "cell-refract-2": {
        "layers": ["voronoi", "refract-post", "derivative-post", "density-map", "saturation"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "refract_range": random.randint(1, 3) * 0.25,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member([m for m in point if m not in point.grid_members()]),
            "voronoi_point_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "refractive voronoi cells, liquid glass, high detail fractal refraction",
            "image_strength": 0.5,
            "cfg_scale": 30,
            "style_preset": "photographic",
        }
    },

    "cell-worms": {
        "layers": ["multires-low", "voronoi", "worms", "density-map", "random-hue", "saturation"],
        "settings": lambda: {
            "freq": random.randint(3, 7),
            "hue_range": 0.125 + random.random() * 0.875,
            "voronoi_alpha": 0.75,
            "voronoi_point_distrib": random_member(point, mask.nonprocedural_members()),
            "voronoi_point_freq": random.randint(2, 4),
            "worms_density": 1500,
            "worms_kink": random.randint(16, 32),
            "worms_stride_deviation": 0,
        },
        "ai": {
            "prompt": "fur, flow field, geometric shapes",
            "image_strength": 0.75,
            "cfg_scale": 15,
        }
    },

    "chalky": {
        "layers": ["basic", "refract-post", "octave-warp-post", "outline", "grain", "lens"],
        "settings": lambda: {
            "color_space": color.oklab,
            "freq": random.randint(2, 3),
            "octaves": random.randint(2, 3),
            "outline_invert": True,
            "refract_range": 0.1 + random.random() * 0.05,
            "ridges": True,
            "warp_octaves": 8,
            "warp_range": 0.0333 + random.random() * 0.016667,
        },
        "ai": {
            "prompt": "scribbles, doodles, style of chalk art",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "chunky-knit": {
        "layers": ["jorts", "random-hue", "contrast-final"],
        "settings": lambda: {
            "angle": random.random() * 360.0,
            "glyph_map_alpha": 0.333 + random.random() * 0.16667,
            "glyph_map_mask": mask.waffle,
            "glyph_map_zoom": 16.0,
        },
        "ai": {
            "prompt": "chunky knit fabric, waffle knit",
            "image_strength": 0.5,
            "cfg_scale": 20,
            "style_preset": "photographic",
        }
    },

    "classic-desktop": {
        "layers": ["basic", "lens-warp"],
        "settings": lambda: {
            "hue_range": 0.333 + random.random() * 0.333,
            "lattice_drift": random.random(),
        },
        "ai": {
            "prompt": "abstract design, art, desktop wallpaper from the era of vintage computing",
        }
    },

    "cloudburst": {
        "layers": ["multires", "reflect-octaves", "octave-warp-octaves", "refract-post",
                   "invert", "grain"],
        "settings": lambda: {
            "color_space": color.hsv,
            "distrib": distrib.exp,
            "freq": 2,
            "hue_range": 0.05 - random.random() * 0.025,
            "hue_rotation": 0.1 - random.random() * 0.025,
            "lattice_drift": 0.75,
            "palette_on": False,
            "reflect_range": 0.125 + random.random() * 0.0625,
            "refract_range": 0.1 + random.random() * 0.05,
            "saturation_distrib": distrib.ones,
            "speed": 0.075,
        },
        "ai": {
            "prompt": "realistic white clouds in a blue sky",
            "image_strength": 0.875,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "clouds": {
        "layers": ["bloom", "grain"],
        "post": lambda settings: [Effect("clouds")],
    },

    "concentric": {
        "layers": ["wobble", "voronoi", "contrast-post", "maybe-palette"],
        "settings": lambda: {
            "color_space": color.rgb,
            "dist_metric": random_member(distance.absolute_members()),
            "distrib": distrib.ones,
            "freq": 2,
            "mask": mask.h_bar,
            "speed": 0.75,
            "spline_order": interp.constant,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_refract": random.randint(8, 16),
            "voronoi_point_drift": 0,
            "voronoi_point_freq": random.randint(1, 2),
        },
        "ai": {
            "prompt": "concentric shape outlines",
        }
    },

    "conference": {
        "layers": ["value-mask", "sobel", "maybe-rotate", "maybe-invert", "grain"],
        "settings": lambda: {
            "freq": 4 * random.randint(6, 12),
            "mask": mask.halftone,
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "design motif, industry conference, truchet pattern",
            "image_strength": 0.925,
            "cfg_scale": 25,
        }
    },

    "contrast-post": {
        "settings": lambda: {
            "contrast_post": 1.25 + random.random() * 0.25
        },
        "post": lambda settings: [Effect("adjust_contrast", amount=settings["contrast_post"])]
    },

    "contrast-final": {
        "settings": lambda: {
            "contrast_final": 1.25 + random.random() * 0.25
        },
        "final": lambda settings: [Effect("adjust_contrast", amount=settings["contrast_final"])]
    },

    "cool-water": {
        "layers": ["basic-water", "funhouse", "bloom", "lens"],
        "settings": lambda: {
            "warp_range": 0.0625 + random.random() * 0.0625,
            "warp_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "complex photorealistic water texture with ripples and light refraction",
            "style_preset": "photographic",
        }
    },

    "corner-case": {
        "layers": ["multires-ridged", "maybe-rotate", "grain", "saturation", "vignette-dark"],
        "settings": lambda: {
            "corners": True,
            "lattice_drift": coin_flip(),
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "generative design, art, right angles, corners"
        }
    },

    "corduroy": {
        "layers": ["jorts", "random-hue", "contrast-final"],
        "settings": lambda: {
            "saturation": 0.625 + random.random() * 0.125,
            "glyph_map_zoom": 8.0,
        },
        "ai": {
            "prompt": "chunky corduroy fabric",
        }
    },

    "cosmic-thread": {
        "layers": ["basic", "worms", "brightness-final", "contrast-final", "bloom"],
        "settings": lambda: {
            "brightness_final": 0.1,
            "color_space": color.rgb,
            "contrast_final": 2.5,
            "worms_alpha": 0.875,
            "worms_behavior": random_member(worms.all()),
            "worms_density": 0.125,
            "worms_drunkenness": 0.125 + random.random() * 0.25,
            "worms_duration": 125,
            "worms_kink": 1.0,
            "worms_stride": 0.75,
            "worms_stride_deviation": 0.0
        },
        "ai": {
            "prompt": "entangled threads, cosmic yarn"
        }
    },

    "cobblestone": {
        "layers": ["bringing-hexy-back", "saturation", "texture", "shadow", "contrast-post", "contrast-final"],
        "settings": lambda: {
            "hue_range": 0.1 + random.random() * 0.05,
            "saturation_final": 0.0 + random.random() * 0.05,
            "shadow_alpha": 0.5,
            "voronoi_point_freq": random.randint(3, 4) * 2,
            "warp_freq": [random.randint(3, 4), random.randint(3, 4)],
            "warp_range": 0.125,
            "warp_octaves": 8
        },
        "ai": {
            "prompt": "cobblestones texture, smooth cobblestone path",
            "image_strength": 0.75,
            "cfg_scale": 20,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "convolution-feedback": {
        "post": lambda settings: [
            Effect("conv_feedback",
                   alpha=.5 * random.random() * 0.25,
                   iterations=random.randint(250, 500)),
        ]
    },

    "corrupt": {
        "post": lambda settings: [
            Effect("warp",
                   displacement=.025 + random.random() * 0.1,
                   freq=[random.randint(2, 4), random.randint(1, 3)],
                   octaves=random.randint(2, 4),
                   spline_order=interp.constant),
        ]
    },

    "crime-scene": {
        "layers": ["value-mask", "maybe-rotate", "grain", "dexter", "dexter", "grime", "lens"],
        "settings": lambda: {
            "mask": mask.chess,
            "mask_repeat": random.randint(2, 3),
            "saturation": 0 if coin_flip() else 0.125,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "checkered floor spattered with red ink, grimey, noir"
        }
    },

    "crooked": {
        "layers": ["starfield", "pixel-sort", "glitchin-out"],
        "settings": lambda: {
            "pixel_sort_angled": True,
            "pixel_sort_darkest": False
        },
        "ai": {
            "prompt": "deep space, space telescope, hst, spitzer, glitch art, pixel sort",
        }
    },

    "crt": {
        "layers": ["scanline-error", "snow"],
        "settings": lambda: {
            "crt_brightness": 0.05,
            "crt_contrast": 1.05,
        },
        "final": lambda settings: [
            Effect("crt"),
            Preset("brightness-final", settings={"brightness_final": settings["crt_brightness"]}),
            Preset("contrast-final", settings={"contrast_final": settings["crt_contrast"]})
        ]
    },

    "crystallize": {
        "layers": ["voronoi", "vignette-bright", "bloom", "contrast-post", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "voronoi_point_freq": 4,
            "voronoi_alpha": 0.875,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 4,
        },
        "ai": {
            "prompt": "stacked cubes, qbert, high detail",
            "style_preset": "isometric"
        }
    },

    "cubert": {
        "layers": ["voronoi", "crt", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "freq": random.randint(4, 6),
            "hue_range": 0.5 + random.random(),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_inverse": True,
            "voronoi_point_distrib": point.h_hex,
            "voronoi_point_freq": random.randint(4, 6),
        },
        "ai": {
            "prompt": "colorful stacked cubes, qbert",
            "style_preset": "isometric"
        }
    },

    "cubic": {
        "layers": ["basic-voronoi", "outline"],
        "settings": lambda: {
            "voronoi_nth": random.randint(2, 8),
            "voronoi_point_distrib": point.concentric,
            "voronoi_point_freq": random.randint(3, 6),
            "voronoi_diagram_type": random_member([voronoi.range, voronoi.color_range]),
        },
        "ai": {
            "prompt": "low-poly, low poly mesh, diffuse lighting",
            "image_strength": 0.5,
            "style_preset": "low-poly",
        }
    },

    "cyclic-dilation": {
        "layers": ["voronoi", "reindex-post", "saturation", "grain"],
        "settings": lambda: {
            "freq": random.randint(24, 48),
            "hue_range": 0.25 + random.random() * 1.25,
            "reindex_range": random.randint(4, 6),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_corners": True,
        },
        "ai": {
            "prompt": "distorted concentric rings"
        }
    },

    "deadbeef": {
        "layers": ["value-mask", "corrupt", "bloom", "crt", "vignette-dark"],
        "settings": lambda: {
            "freq": 6 * random.randint(9, 24),
            "mask": mask.alphanum_hex,
        },
        "ai": {
            "prompt": "a glitched hexadecimal display, completely broken computer code"
        }
    },

    "death-star-plans": {
        "layers": ["voronoi", "refract-post", "maybe-rotate", "posterize", "sobel", "invert", "crt", "vignette-dark"],
        "settings": lambda: {
            "dist_metric": random_member([distance.chebyshev, distance.manhattan]),
            "posterize_levels": random.randint(3, 4),
            "refract_range": 0.5 + random.random() * 0.25,
            "refract_y_from_offset": True,
            "voronoi_alpha": 1,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": random.randint(1, 3),
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "complicated blueprint design, map of the death star, sci-fi hologram, display scanlines",
        }
    },

    "deep-field": {
        "layers": ["multires", "refract-octaves", "octave-warp-octaves", "bloom", "lens"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "freq": random.randint(8, 10),
            "hue_range": 1,
            "mask": mask.sparser,
            "mask_static": True,
            "lattice_drift": 1,
            "octave_blending": blend.alpha,
            "octaves": 5,
            "palette_on": False,
            "speed": 0.05,
            "refract_range": 0.2 + random.random() * 0.1,
            "warp_freq": 2,
            "warp_signed_range": True,
        },
        "ai": {
            "prompt": "hubble space telescope, hst, spitzer space telescope, deep field, galaxies",
            "image_strength": 0.75,
            "cfg_scale": 20,
            "style_preset": "photographic",
        }
    },

    "deeper": {
        "layers": ["multires-alpha", "funhouse", "lens"],
        "settings": lambda: {
            "hue_range": 0.75,
            "octaves": 6,
            "ridges": True,
        },
        "ai": {
            "prompt": "high detail, thread, yarn, fiber, fabric, string theory",
        }
    },

    "degauss": {
        "final": lambda settings: [
            Effect("degauss", displacement=.06 + random.random() * 0.03),
            Preset("crt"),
        ]
    },

    "density-map": {
        "layers": ["grain"],
        "post": lambda settings: [Effect("density_map"), Effect("convolve", kernel=mask.conv2d_invert)],
    },

    "density-wave": {
        "layers": [random_member(["basic", "symmetry"]), "reflect-post", "density-map", "invert", "bloom"],
        "settings": lambda: {
            "reflect_range": random.randint(3, 8),
            "saturation": random.randint(0, 1),
        },
        "ai": {
            "prompt": "density plot, wave",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "derivative-octaves": {
        "settings": lambda: {
            "deriv_alpha": 1.0,
            "dist_metric": random_member(distance.absolute_members())
        },
        "octaves": lambda settings: [
            Effect("derivative", dist_metric=settings["dist_metric"], alpha=settings["deriv_alpha"])
        ]
    },

    "derivative-post": {
        "settings": lambda: {
            "deriv_alpha": 1.0,
            "dist_metric": random_member(distance.absolute_members())
        },
        "post": lambda settings: [
            Effect("derivative", dist_metric=settings["dist_metric"], alpha=settings["deriv_alpha"])
        ]
    },

    "dexter": {
        "layers": ["spatter-final"],
        "settings": lambda: {
            "spatter_final_color": [.35 + random.random() * 0.15,
                                    0.025 + random.random() * 0.0125,
                                    0.075 + random.random() * 0.0375],
        },
    },

    "different": {
        "layers": ["multires", "sine-octaves", "reflect-octaves", "reindex-octaves", "funhouse", "lens"],
        "settings": lambda: {
            "freq": [random.randint(4, 6), random.randint(4, 6)],
            "reflect_range": 7.5 + random.random() * 5.0,
            "reindex_range": 0.25 + random.random() * 0.25,
            "sine_range": random.randint(7, 12),
            "speed": 0.025,
            "warp_range": 0.0375 * random.random() * 0.0375,
        },
        "ai": {
            "prompt": "quirky psychedelic texture",
        }
    },

    "distressed": {
        "layers": ["grain", "filthy", "saturation"],
    },

    "distance": {
        "layers": ["multires", "derivative-octaves", "bloom", "shadow", "contrast-final", "maybe-rotate", "lens"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "distrib": distrib.exp,
            "freq": [random.randint(4, 5), random.randint(2, 3)],
            "lattice_drift": 1,
            "saturation": 0.0625 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "distance, distant mood, planet from orbit, atmospheric cloud cover, surface from high orbit",
        }
    },

    "dla": {
        "layers": ["basic", "contrast-final"],
        "settings": lambda: {
            "dla_alpha": 0.666 + random.random() * 0.333,
            "dla_padding": random.randint(2, 8),
            "dla_seed_density": 0.2 + random.random() * 0.1,
            "dla_density": 0.1 + random.random() * 0.05,
        },
        "post": lambda settings: [
            Effect("dla",
                   alpha=settings["dla_alpha"],
                   padding=settings["dla_padding"],
                   seed_density=settings["dla_seed_density"],
                   density=settings["dla_density"])
        ],
        "ai": {
            "prompt": "diffusion-limited aggregation, lichtenberg figure, electrical discharge, branching out, branched structure",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "dla-forest": {
        "layers": ["dla", "reverb", "contrast-final", "bloom"],
        "settings": lambda: {
            "dla_padding": random.randint(2, 8),
            "reverb_iterations": random.randint(2, 4),
        },
        "ai": {
            "prompt": "diffusion-limited aggregation, lichtenberg figure, electrical discharge, branching out, branched structure, tributaries, capillaries",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "domain-warp": {
        "layers": ["multires-ridged", "refract-post", "vaseline", "grain", "vignette-dark", "saturation"],
        "settings": lambda: {
            "refract_range": 0.5 + random.random() * 0.5,
        },
        "ai": {
            "prompt": "domain warping example, fractional brownian motion",
            "image_strength": 0.75,
            "cfg_scale": 10,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "dropout": {
        "layers": ["basic", "derivative-post", "maybe-invert", "grain"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "distrib": distrib.ones,
            "freq": [random.randint(4, 6), random.randint(2, 4)],
            "mask": mask.dropout,
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(4, 6),
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "classic procedural art, outlines of overlapping squares, lots of right angles"
        }
    },

    "eat-static": {
        "layers": ["basic", "be-kind-rewind", "scanline-error", "crt"],
        "settings": lambda: {
            "freq": 512,
            "saturation": 0,
            "speed": 2.0,
        },
        "ai": {
            "prompt": "a screen full of static",
        }
    },

    "educational-video-film": {
        "layers": ["basic", "be-kind-rewind"],
        "settings": lambda: {
            "color_space": color.oklab,
            "ridges": True,
        },
        "ai": {
            "prompt": "a colorful intro to a 1980s educational vhs film"
        }
    },

    "electric-worms": {
        "layers": ["voronoi", "worms", "density-map", "glowing-edges", "bloom"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.octagram, distance.triangular]),
            "freq": random.randint(3, 6),
            "lattice_drift": 1,
            "voronoi_alpha": 0.25 + random.random() * 0.25,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(0, 3),
            "voronoi_point_freq": random.randint(3, 6),
            "voronoi_point_distrib": point.random,
            "worms_alpha": 0.666 + random.random() * 0.333,
            "worms_behavior": worms.random,
            "worms_density": 1000,
            "worms_duration": 1,
            "worms_kink": random.randint(7, 9),
            "worms_stride_deviation": 16,
        },
        "ai": {
            "prompt": "electric worms, electrified flow field, psychedelic fractal art",
            "image_strength": .75,
            "cfg_scale": 15,
        }
    },

    "emboss": {
        "post": lambda settings: [Effect("convolve", kernel=mask.conv2d_emboss)]
    },

    "emo": {
        "layers": ["value-mask", "voronoi", "contrast-final", "maybe-rotate", "saturation", "tint", "lens"],
        "settings": lambda: {
            "contrast_final": 4.0,
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "mask": mask.emoji,
            "spline_order": interp.cosine,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_refract": 0.125 + random.random() * 0.25,
        },
        "ai": {
            "prompt": "black and white design with distorted symbols and geometric shapes",
        }
    },

    "emu": {
        "layers": ["value-mask", "voronoi", "saturation", "distressed"],
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "distrib": distrib.ones,
            "mask": stash("mask", random_member(enum_range(mask.emoji_00, mask.emoji_26))),
            "mask_repeat": 1,
            "spline_order": interp.constant,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_point_distrib": stash("mask"),
            "voronoi_refract": 0.125 + random.random() * 0.125,
            "voronoi_refract_y_from_offset": False,
        },
        "ai": {
            "prompt": "black and white design with distorted symbols and geometric shapes",
        }
    },

    "entities": {
        "layers": ["value-mask", "refract-octaves", "normalize"],
        "settings": lambda: {
            "hue_range": 2.0 + random.random() * 2.0,
            "mask": mask.invaders_square,
            "mask_repeat": random.randint(3, 4) * 2,
            "refract_range": 0.1 + random.random() * 0.05,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "a grid of entities, pantheon of mayan gods, psychedelic visionary art, ancient language",
            "image_strength": 0.75,
            "cfg_scale": 25,
        },
    },

    "entity": {
        "layers": ["entities", "sobel", "invert", "bloom", "random-hue", "lens"],
        "settings": lambda: {
            "corners": True,
            "distrib": distrib.ones,
            "hue_range": 1.0 + random.random() * 0.5,
            "mask_repeat": 1,
            "refract_range": 0.025 + random.random() * 0.0125,
            "refract_signed_range": True,
            "refract_y_from_offset": False,
            "speed": 0.05,
        },
        "ai": {
            "prompt": "a single large entity, avatar of a mysterious god, symmetrical, psychedelic visionary art",
        },
    },

    "erosion-worms": {
        "settings": lambda: {
            "erosion_worms_alpha": 0.5 + random.random() * 0.5,
            "erosion_worms_contraction": 0.5 + random.random() * 0.5,
            "erosion_worms_density": random.randint(25, 100),
            "erosion_worms_inverse": False,
            "erosion_worms_iterations": random.randint(25, 100),
            "erosion_worms_xy_blend": 0.75 + random.random() * 0.25
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
        ],
        "ai": {
            "prompt": "flow field, lines tracing the path of erosion",
        }
    },

    "escape-velocity": {
        "layers": ["multires-low", "erosion-worms", "lens"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "distrib": random_member([distrib.exp, distrib.uniform]),
            "erosion_worms_contraction": 0.2 + random.random() * 0.1,
            "erosion_worms_iterations": random.randint(625, 1125),
        },
        "ai": {
            "prompt": "flow field, path of orbiting subatomic particles escaping gravitational pull, particle detector, ionizing radiation, microsingularities, baby black holes, microscopic wormholes, relativistic jets, quantum gravity",
        }
    },

    "falsetto": {
        "final": lambda settings: [Effect("false_color")]
    },

    "fargate": {
        "layers": ["serene", "contrast-post", "crt", "saturation"],
        "settings": lambda: {
            "brightness_distrib": distrib.uniform,
            "freq": 3,
            "octaves": 3,
            "refract_range": 0.015 + random.random() * 0.0075,
            "saturation_distrib": distrib.uniform,
            "speed": -0.25,
            "value_distrib": distrib.center_circle,
            "value_freq": 3,
            "value_refract_range": 0.015 + random.random() * 0.0075,
        },
        "ai": {
            "prompt": "outward ripples, serene, peaceful",
        }
    },

    "fast-eddies": {
        "layers": ["basic", "voronoi", "worms", "contrast-final", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "hue_range": 0.25 + random.random() * 0.75,
            "hue_rotation": random.random(),
            "octaves": random.randint(1, 3),
            "palette_on": False,
            "ridges": coin_flip(),
            "voronoi_alpha": 0.5 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": random.randint(2, 6),
            "voronoi_refract": 1.0,
            "worms_alpha": 0.5 + random.random() * 0.5,
            "worms_behavior": worms.chaotic,
            "worms_density": 1000,
            "worms_duration": 6,
            "worms_kink": random.randint(125, 375),
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "ai": {
            "prompt": "psychedelic fractal turbulence, swirling, unmixed, eddies, flow field, fractal flame",
            "image_strength": 0.625,
            "cfg_scale": 25,
            "style_preset": "photographic",
        },
    },

    "fibers": {
        "final": lambda settings: [Effect("fibers")]
    },

    "figments": {
        "layers": ["multires-low", "voronoi", "funhouse", "wormhole", "bloom", "contrast-final", "lens"],
        "settings": lambda: {
            "freq": 2,
            "hue_range": 2,
            "lattice_drift": 1,
            "speed": 0.025,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_refract": 0.333 + random.random() * 0.333,
            "wormhole_stride": 0.02 + random.random() * 0.01,
            "wormhole_kink": 4,
        },
        "ai": {
            "prompt": "psychedelic fractal turbulence, swirling, unmixed, eddies",
        },
    },

    "filthy": {
        "layers": ["grime", "scratches", "stray-hair"],
    },

    "fireball": {
        "layers": ["basic", "periodic-refract", "refract-post", "refract-post", "bloom", "lens", "contrast-final"],
        "settings": lambda: {
            "contrast_final": 2.5,
            "distrib": distrib.center_circle,
            "hue_rotation": 0.925,
            "freq": 1,
            "refract_range": 0.025 + random.random() * 0.0125,
            "refract_y_from_offset": False,
            "value_distrib": distrib.center_circle,
            "value_freq": 1,
            "value_refract_range": 0.05 + random.random() * 0.025,
            "speed": 0.05,
        },
        "ai": {
            "prompt": "a swirling ball of flame, fireball"
        }
    },

    "financial-district": {
        "layers": ["voronoi", "bloom", "contrast-final", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.manhattan,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_point_distrib": point.random,
            "voronoi_nth": random.randint(1, 3),
            "voronoi_point_freq": 2,
        },
        "ai": {
            "prompt": "financial district, financial institution motif",
        }
    },

    "fossil-hunt": {
        "layers": ["voronoi", "refract-octaves", "posterize-outline", "grain", "saturation"],
        "settings": lambda: {
            "freq": random.randint(3, 5),
            "lattice_drift": 1.0,
            "posterize_levels": random.randint(3, 5),
            "refract_range": random.randint(2, 4) * 0.5,
            "refract_y_from_offset": True,
            "voronoi_alpha": 0.5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_freq": 10,
        },
        "ai": {
            "prompt": "buried fossils, paleontology, layered rock, plants, animals, fungi, bacteria",
            "image_strength": 0.625,
            "cfg_scale": 30,
            "style_preset": "photographic"
        }
    },

    "fractal-forms": {
        "layers": ["fractal-seed"],
        "settings": lambda: {
            "worms_kink": random.randint(256, 512),
        },
        "ai": {
            "prompt": "psychedelic fractal turbulence, swirling, unmixed, eddies",
            "image_strength": 0.625,
            "cfg_scale": 25,
        },
    },

    "fractal-seed": {
        "layers": ["multires-low", "worms", "density-map", "random-hue", "bloom", "shadow",
                   "contrast-final", "saturation", "aberration"],
        "settings": lambda: {
            "freq": random.randint(2, 3),
            "hue_range": 1.0 + random.random() * 3.0,
            "ridges": coin_flip(),
            "speed": 0.05,
            "palette_on": False,
            "worms_behavior": random_member([worms.chaotic, worms.random]),
            "worms_alpha": 0.9 + random.random() * 0.1,
            "worms_density": random.randint(750, 1250),
            "worms_duration": random.randint(2, 3),
            "worms_kink": 1.0,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "ai": {
            "prompt": "psychedelic fractal turbulence, swirling, unmixed, eddies",
            "image_strength": 0.625,
            "cfg_scale": 25,
        },
    },

    "fractal-smoke": {
        "layers": ["fractal-seed"],
        "settings": lambda: {
            "worms_behavior": worms.random,
            "worms_stride": random.randint(96, 192),
        },
        "ai": {
            "prompt": "smoke, psychedelic fractal turbulence, swirling, unmixed, eddies",
            "image_strength": 0.625,
            "cfg_scale": 25,
        },
    },

    "fractile": {
        "layers": ["symmetry", "voronoi", "reverb", "contrast-post", "palette", "random-hue",
                   "maybe-rotate", "lens"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "reverb_iterations": random.randint(2, 4),
            "reverb_octaves": random.randint(2, 4),
            "voronoi_alpha": 0.5 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(0, 2),
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "tile flooring, shower tiles, symmetrical decorative design, marble, granite, ceramic",
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        },
    },

    "fundamentals": {
        "layers": ["voronoi", "derivative-post", "density-map", "grain", "saturation"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "freq": random.randint(3, 5),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": random.randint(3, 5),
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_refract": 0.125 + random.random() * 0.0625,
        },
        "ai": {
            "prompt": "the cover design for a computer science textbook",
        },
    },

    "funhouse": {
        "settings": lambda: {
            "warp_freq": [random.randint(2, 4), random.randint(2, 4)],
            "warp_octaves": random.randint(1, 4),
            "warp_range": 0.25 + random.random() * 0.125,
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
        "layers": ["value-mask", "refract-post", "contrast-final", "maybe-rotate", "saturation", "lens", "grain"],
        "settings": lambda: {
            "distrib": random_member([distrib.ones, distrib.uniform]),
            "mask": random_member(mask.glyph_members()),
            "mask_repeat": random.randint(1, 6),
            "octaves": random.randint(1, 2),
            "refract_range": 0.125 + random.random() * 0.125,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        },
        "ai": {
            "prompt": "stylized distorted symbols"
        }
    },

    "galalaga": {
        "layers": ["value-mask", "contrast-final", "glitchin-out"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "hue_range": random.random() * 2.5,
            "mask": mask.invaders_square,
            "mask_repeat": 4,
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
        ],
        "ai": {
            "prompt": "classic video game, galaga, galaxian, space invaders, centipede",
            "style_preset": "pixel-art",
        },
    },

    "game-show": {
        "layers": ["basic", "posterize", "be-kind-rewind"],
        "settings": lambda: {
            "freq": random.randint(8, 16) * 2,
            "mask": random_member([mask.h_tri, mask.v_tri]),
            "posterize_levels": random.randint(2, 5),
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "background art from a 1960s game show, sixties, retro",
        }
    },

    "glacial": {
        "layers": ["fractal-smoke"],
        "settings": lambda: {
            "worms_quantize": True,
        },
        "ai": {
            "prompt": "satellite imagery of melting glaciers, melted icecaps flowing into the ocean, icebergs, global warming, greenland, antarctica",
            "style_preset": "photographic",
        }
    },

    "glitchin-out": {
        "layers": ["corrupt"],
        "final": lambda settings: [Effect("glitch"), Preset("crt"), Preset("bloom")]
    },

    "globules": {
        "layers": ["multires-low", "reflect-octaves", "density-map", "shadow", "lens"],
        "settings": lambda: {
            "distrib": distrib.ones,
            "freq": random.randint(3, 6),
            "hue_range": 0.25 + random.random() * 0.5,
            "lattice_drift": 1,
            "mask": mask.sparse,
            "mask_static": True,
            "octaves": random.randint(3, 6),
            "palette_on": False,
            "reflect_range": 2.5,
            "saturation": 0.175 + random.random() * 0.175,
            "speed": 0.125,
        },
        "ai": {
            "prompt": "deep space nebula imagery, dark globules, dark matter, stellar nursery",
        }
    },

    "glom": {
        "layers": ["basic", "refract-octaves", "reflect-octaves", "refract-post", "reflect-post", "funhouse",
                   "bloom", "shadow", "contrast-post", "lens"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "freq": [2, 2],
            "hue_range": 0.25 + random.random() * 0.125,
            "lattice_drift": 1,
            "octaves": 2,
            "reflect_range": 0.625 + random.random() * 0.375,
            "refract_range": 0.333 + random.random() * 0.16667,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "speed": 0.025,
            "warp_range": 0.0625 + random.random() * 0.030625,
            "warp_octaves": 1,
        },
        "ai": {
            "prompt": "a gooey sticky dark liquid oozing around, gel, gelatinous ooze",
        }
    },

    "glowing-edges": {
        "final": lambda settings: [Effect("glowing_edges")]
    },

    "glyph-map": {
        "layers": ["basic"],
        "settings": lambda: {
            "glyph_map_alpha": 1.0,
            "glyph_map_colorize": coin_flip(),
            "glyph_map_spline_order": interp.constant,
            "glyph_map_mask": random_member(set(mask.procedural_members()).intersection(masks.square_masks())),
            "glyph_map_zoom": random.randint(6, 10),
        },
        "post": lambda settings: [
            Effect("glyph_map",
                   alpha=settings["glyph_map_alpha"],
                   colorize=settings["glyph_map_colorize"],
                   mask=settings["glyph_map_mask"],
                   spline_order=settings["glyph_map_spline_order"],
                   zoom=settings["glyph_map_zoom"])
        ],
        "ai": {
            "prompt": "a grid of glyphs and symbols, truchet pattern",
            "image_strength": 0.925,
            "cfg_scale": 15,
        }
    },

    "glyphic": {
        "layers": ["value-mask", "posterize", "palette", "saturation", "maybe-invert", "distressed"],
        "settings": lambda: {
            "corners": True,
            "mask": random_member(mask.procedural_members()),
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(3, 5),
            "posterize_levels": 1,
            "saturation": 0,
            "spline_order": interp.cosine,
        },
        "generator": lambda settings: {
            "freq": masks.mask_shape(settings["mask"])[0:2],
        },
        "ai": {
            "prompt": "stylized distorted symbols, glyphs, truchet pattern",
        }
    },

    "grain": {
        "unique": True,
        "settings": lambda: {
            "grain_alpha": 0.0333 + random.random() * 0.01666,
            "grain_brightness": 0.0125 + random.random() * 0.00625,
            "grain_contrast": 1.025 + random.random() * 0.0125
        },
        "final": lambda settings: [
            Effect("grain", alpha=settings["grain_alpha"]),
            Preset("brightness-final", settings={"brightness_final": settings["grain_brightness"]}),
            Preset("contrast-final", settings={"contrast_final": settings["grain_contrast"]})
        ]
    },

    "graph-paper": {
        "layers": ["wobble", "voronoi", "derivative-post", "maybe-rotate", "lens", "crt", "bloom", "contrast-final"],
        "settings": lambda: {
            "color_space": color.rgb,
            "corners": True,
            "distrib": distrib.ones,
            "dist_metric": distance.euclidean,
            "freq": random.randint(3, 4) * 2,
            "mask": mask.chess,
            "spline_order": interp.constant,
            "voronoi_alpha": 0.5 + random.random() * 0.25,
            "voronoi_refract": 0.75 + random.random() * 0.375,
            "voronoi_refract_y_from_offset": True,
            "voronoi_diagram_type": voronoi.flow,
        },
        "ai": {
            "prompt": "wireframe terrain on a grid, radar, lidar, retro digital map, vintage computing, vector",
        }
    },

    "grass": {
        "layers": ["multires", "worms", "grain"],
        "settings": lambda: {
            "color_space": color.hsv,
            "freq": random.randint(6, 12),
            "hue_rotation": 0.25 + random.random() * 0.05,
            "lattice_drift": 1,
            "palette_on": False,
            "saturation": 0.625 + random.random() * 0.25,
            "worms_behavior": random_member([worms.chaotic, worms.meandering]),
            "worms_alpha": 0.9,
            "worms_density": 50 + random.random() * 25,
            "worms_drunkenness": 0.125,
            "worms_duration": 1.125,
            "worms_stride": 0.875,
            "worms_stride_deviation": 0.125,
            "worms_kink": 0.125 + random.random() * 0.5,
        },
        "ai": {
            "prompt": "grassy texture, grass and dirt, multi-colored grass, thatch, a dusty old lawn, turf",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "tile-texture",
        }
    },

    "grayscale": {
        "final": lambda settings: [Effect("adjust_saturation", amount=0)]
    },

    "griddy": {
        "layers": ["basic", "sobel", "invert", "bloom"],
        "settings": lambda: {
            "freq": random.randint(3, 9),
            "mask": mask.chess,
            "octaves": random.randint(3, 8),
            "spline_order": interp.constant
        },
        "ai": {
            "prompt": "a grid within a grid, recursive grids",
            "image_strength": .5,
            "cfg_scale": 20,
        }
    },

    "grime": {
        "final": lambda settings: [Effect("grime")]
    },

    "groove-is-stored-in-the-heart": {
        "layers": ["basic", "posterize", "ripple", "distressed"],
        "settings": lambda: {
            "distrib": distrib.column_index,
            "posterize_levels": random.randint(1, 2),
            "ripple_range": 0.75 + random.random() * 0.375,
        },
        "ai": {
            "prompt": "groovy, psychedelia, peace and love, groove is stored in the heart, vintage",
            "image_strength": 0.95,
            "cfg_scale": 25,
        }
    },

    "halt-catch-fire": {
        "layers": ["multires-low", "pixel-sort", "maybe-rotate", "glitchin-out"],
        "settings": lambda: {
            "freq": 2,
            "hue_range": 0.05,
            "lattice_drift": 1,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "glitchy digital art, halt and catch fire, corrupted jpeg",
        }
    },

    "hearts": {
        "layers": ["value-mask", "skew", "posterize", "crt"],
        "settings": lambda: {
            "distrib": distrib.ones,
            "hue_distrib": None if coin_flip() else random_member([distrib.column_index, distrib.row_index]),
            "hue_rotation": 0.925,
            "mask": mask.mcpaint_19,
            "mask_repeat": random.randint(8, 12),
            "posterize_levels": random.randint(1, 2),
        },
        "ai": {
            "prompt": "stylized heart symbols, unicode pixel font",
            "style_preset": "pixel-art",
        }
    },

    "hotel-carpet": {
        "layers": ["basic", "ripple", "carpet", "grain"],
        "settings": lambda: {
            "ripple_kink": 0.5 + random.random() * 0.25,
            "ripple_range": 0.666 + random.random() * 0.333,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "hotel carpet, psychedelia, groovy 1960s design",
        }
    },

    "hsv-gradient": {
        "layers": ["basic", "maybe-rotate", "grain", "saturation"],
        "settings": lambda: {
            "color_space": color.hsv,
            "hue_range": 0.5 + random.random() * 2.0,
            "lattice_drift": 1.0,
            "palette_on": False,
        },
        "ai": {
            "prompt": "color gradient, color wash, hue saturation brightness",
        }
    },

    "hydraulic-flow": {
        "layers": ["multires", "derivative-octaves", "refract-octaves", "erosion-worms", "density-map",
                   "maybe-invert", "shadow", "bloom", "maybe-rotate", "lens"],
        "settings": lambda: {
            "deriv_alpha": 0.25 + random.random() * 0.25,
            "erosion_worms_alpha": 0.125 + random.random() * 0.125,
            "erosion_worms_contraction": 0.75 + random.random() * 0.5,
            "erosion_worms_density": random.randint(5, 250),
            "erosion_worms_iterations": random.randint(50, 250),
            "freq": 2,
            "hue_range": random.random(),
            "palette_on": False,
            "refract_range": random.random(),
            "ridges": coin_flip(),
            "saturation": random.random(),
        },
        "ai": {
            "prompt": "hydraulic flow, erosion, flow field",
        }
    },

    "i-made-an-art": {
        "layers": ["basic", "outline", "distressed", "contrast-final", "saturation"],
        "settings": lambda: {
            "spline_order": interp.constant,
            "lattice_drift": random.randint(5, 10),
            "hue_range": random.random() * 4,
            "hue_rotation": random.random(),
        },
        "ai": {
            "prompt": "modern art, mondrian, squares, colors",
        }
    },

    "inkling": {
        "layers": ["voronoi", "refract-post", "funhouse", "grayscale", "density-map", "contrast-post",
                   "maybe-invert", "fibers", "grime", "scratches"],
        "settings": lambda: {
            "distrib": distrib.ones,
            "dist_metric": distance.euclidean,
            "contrast_post": 2.5,
            "freq": random.randint(2, 4),
            "lattice_drift": 1.0,
            "mask": mask.dropout,
            "mask_static": True,
            "refract_range": 0.25 + random.random() * 0.125,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_refract": 0.25 + random.random() * 0.125,
            "warp_range": 0.125 + random.random() * 0.0625,
        },
        "ai": {
            "prompt": "spilled ink on paper, unmixed paint",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "invert": {
        "post": lambda settings: [Effect("convolve", kernel=mask.conv2d_invert)]
    },

    "is-this-anything": {
        "layers": ["soup"],
        "settings": lambda: {
            "refract_range": 2.5 + random.random() * 1.25,
            "voronoi_point_freq": 1,
        },
        "ai": {
            "prompt": "fractal flame, singularity, flow field, soupy",
        }
    },

    "its-the-fuzz": {
        "layers": ["multires-low", "muppet-fur"],
        "settings": lambda: {
            "worms_behavior": worms.unruly,
            "worms_drunkenness": 0.5 + random.random() * 0.25,
            "worms_duration": 2.0 + random.random(),
        },
        "ai": {
            "prompt": "colorful fuzz, felt, fibers, fabric, lint, wool, flow field",
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "jorts": {
        "layers": ["glyph-map", "funhouse", "skew", "shadow", "brightness-post", "contrast-post", "vignette-dark",
                   "grain", "saturation"],
        "settings": lambda: {
            "angle": 0,
            "freq": [128, 512],
            "glyph_map_alpha": 0.5 + random.random() * 0.25,
            "glyph_map_colorize": True,
            "glyph_map_mask": mask.v_bar,
            "glyph_map_spline_order": interp.linear,
            "glyph_map_zoom": 4.0,
            "hue_rotation": 0.5 + random.random() * 0.05,
            "hue_range": 0.0625 + random.random() * 0.0625,
            "palette_on": False,
            "warp_freq": [random.randint(2, 3), random.randint(2, 3)],
            "warp_range": 0.0075 + random.random() * 0.00625,
            "warp_octaves": 1,
        },
        "ai": {
            "prompt": "denim fabric texture, corduroy, blue jeans, levis",
            "cfg_scale": 20,
            "style_preset": "tile-texture",
        }
    },

    "jovian-clouds": {
        "layers": ["voronoi", "worms", "brightness-post", "contrast-post", "shadow", "tint", "grain", "saturation",
                   "lens"],
        "settings": lambda: {
            "contrast_post": 2.0,
            "dist_metric": distance.euclidean,
            "freq": [random.randint(4, 7), random.randint(1, 3)],
            "hue_range": 0.333 + random.random() * 0.16667,
            "hue_rotation": 0.5,
            "voronoi_alpha": 0.175 + random.random() * 0.25,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(8, 10),
            "voronoi_refract": 5.0 + random.random() * 3.0,
            "worms_behavior": worms.chaotic,
            "worms_alpha": 0.175 + random.random() * 0.25,
            "worms_density": 500,
            "worms_duration": 2.0,
            "worms_kink": 192,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0625,
        },
        "ai": {
            "prompt": "great red spot of jupiter, jovian clouds, gas giant atmosphere, juno satellite imagery, storms, hurricanes, cyclones, jetstream, swirling, unmixed, eddies",
            "image_strength": 0.35,
            "cfg_scale": 25,
            "style_preset": "photographic",
        },
    },

    "just-refracts-maam": {
        "layers": ["basic", "refract-octaves", "refract-post", "shadow", "lens"],
        "settings": lambda: {
            "corners": True,
            "refract_range": 0.5 + random.random() * 0.5,
            "ridges": coin_flip(),
        },
        "ai": {
            "prompt": "distorted glass refracting a psychedelic fractal pattern",
            "image_strength": 0.375,
            "cfg_scale": 25,
            "style_preset": "photographic"
        }
    },

    "kaleido": {
        "layers": ["voronoi-refract", "wobble"],
        "settings": lambda: {
            "color_space": color.hsv,
            "freq": random.randint(8, 12),
            "hue_range": .5 + random.random() * 2.5,
            "kaleido_point_corners": True,
            "kaleido_point_distrib": point.random,
            "kaleido_point_freq": 1,
            "kaleido_sdf_sides": random.randint(0, 10),
            "kaleido_sides": random.randint(3, 16),
            "kaleido_blend_edges": False,
            "palette_on": False,
            "speed": 0.125,
            "voronoi_point_freq": random.randint(8, 12),
        },
        "post": lambda settings: [
            Effect("kaleido",
                   blend_edges=settings["kaleido_blend_edges"],
                   point_corners=settings["kaleido_point_corners"],
                   point_distrib=settings["kaleido_point_distrib"],
                   point_freq=settings["kaleido_point_freq"],
                   sdf_sides=settings["kaleido_sdf_sides"],
                   sides=settings["kaleido_sides"]),
        ],
        "ai": {
            "prompt": "abstract psychedelic fractal pattern",
            "image_strength": 0.05,
            "cfg_scale": 35,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "knotty-clouds": {
        "layers": ["basic", "voronoi", "worms"],
        "settings": lambda: {
            "voronoi_alpha": 0.125 + random.random() * 0.25,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_freq": random.randint(6, 10),
            "worms_alpha": 0.666 + random.random() * 0.333,
            "worms_behavior": worms.obedient,
            "worms_density": 1000,
            "worms_duration": 1,
            "worms_kink": 4,
        },
        "ai": {
            "prompt": "flow field",
            "image_strength": 0.95,
            "cfg_scale": 25,
        }
    },

    "later": {
        "layers": ["value-mask", "multires", "wobble", "voronoi", "funhouse", "glowing-edges", "crt", "vignette-dark"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "freq": random.randint(4, 8),
            "mask": random_member(mask.procedural_members()),
            "spline_order": interp.constant,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(4, 8),
            "voronoi_refract": 2.0 + random.random(),
            "warp_freq": random.randint(2, 4),
            "warp_spline_order": interp.bicubic,
            "warp_octaves": 2,
            "warp_range": 0.05 + random.random() * 0.025,
        },
        "ai": {
            "prompt": "neon blacklight psychedelic fractal design",
        }
    },

    "lattice-noise": {
        "layers": ["basic", "derivative-octaves", "derivative-post", "density-map", "shadow",
                   "grain", "saturation", "vignette-dark"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "freq": random.randint(2, 5),
            "lattice_drift": 1.0,
            "octaves": random.randint(2, 3),
            "ridges": coin_flip(),
        },
        "ai": {
            "prompt": "distorted psychedelic grid, deformed lattice",
            "image_strength": 0.975,
            "cfg_scale": 30,
        }
    },

    "lcd": {
        "layers": ["value-mask", "invert", "skew", "shadow", "vignette-bright", "grain"],
        "settings": lambda: {
            "mask": random_member([mask.lcd, mask.lcd_binary]),
            "mask_repeat": random.randint(8, 12),
            "saturation": 0.0,
        },
        "ai": {
            "prompt": "alphanumeric lcd display",
            "style_preset": "photographic"
        }
    },

    "lens": {
        "layers": ["lens-distortion", "aberration", "vaseline", "tint", "vignette-dark"],
        "settings": lambda: {
            "lens_brightness": 0.05 + random.random() * 0.025,
            "lens_contrast": 1.05 + random.random() * 0.025
        },
        "final": lambda settings: [
            Preset("brightness-final", settings={"brightness_final": settings["lens_brightness"]}),
            Preset("contrast-final", settings={"contrast_final": settings["lens_contrast"]})
        ]
    },

    "lens-distortion": {
        "final": lambda settings: [
            Effect("lens_distortion", displacement=(.125 + random.random() * 0.0625) * (1 if coin_flip() else -1)),
        ]
    },

    "lens-warp": {
        "post": lambda settings: [
            Effect("lens_warp", displacement=.125 + random.random() * 0.0625),
            Effect("lens_distortion", displacement=.25 + random.random() * 0.125 * (1 if coin_flip() else -1)),
        ]
    },

    "light-leak": {
        "layers": ["vignette-bright"],
        "final": lambda settings: [Effect("light_leak", alpha=.125 + random.random() * 0.0625), Preset("bloom")]
    },

    "look-up": {
        "layers": ["multires-alpha", "brightness-post", "contrast-post", "contrast-final", "saturation", "lens", "bloom"],
        "settings": lambda: {
            "brightness_post": -0.075,
            "color_space": color.hsv,
            "contrast_final": 1.5,
            "distrib": distrib.exp,
            "freq": random.randint(30, 40),
            "hue_range": 0.333 + random.random() * 0.333,
            "lattice_drift": 0,
            "mask": mask.sparsest,
            "octaves": 10,
            "ridges": True,
            "saturation": 0.5,
            "speed": 0.025,
        },
        "ai": {
            "prompt": "night sky, stars, milky way, lens flare, stargazing",
            "image_strength": 0.625,
            "cfg_scale": 20,
            "style_preset": "photographic",
        }
    },

    "low-poly": {
        "settings": lambda: {
            "lowpoly_distrib": random_member(point.circular_members()),
            "lowpoly_freq": random.randint(10, 20),
        },
        "post": lambda settings: [
            Effect("lowpoly",
                   distrib=settings["lowpoly_distrib"],
                   freq=settings["lowpoly_freq"])
        ],
    },

    "low-poly-regions": {
        "layers": ["voronoi", "low-poly"],
        "settings": lambda: {
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_point_freq": random.randint(2, 3),
        },
        "ai": {
            "prompt": "low-poly mesh, diffuse lighting",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "low-poly",
        }
    },

    "lsd": {
        "layers": ["basic", "refract-post", "invert", "random-hue", "lens", "grain"],
        "settings": lambda: {
            "brightness_distrib": distrib.ones,
            "freq": random.randint(3, 4),
            "hue_range": random.randint(3, 4),
            "speed": 0.025,
        },
        "ai": {
            "prompt": "psychedelic fractal artwork, swirling, unmixed, trippy",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "magic-smoke": {
        "layers": ["multires", "worms", "lens"],
        "settings": lambda: {
            "octaves": random.randint(2, 3),
            "worms_alpha": 1,
            "worms_behavior": random_member([worms.obedient, worms.crosshatch]),
            "worms_density": 750,
            "worms_duration": 0.25,
            "worms_kink": random.randint(1, 3),
            "worms_stride": random.randint(64, 256),
        },
        "ai": {
            "prompt": "smoke rising, magic smoke, wisps of smoke wafting in the air, wispy",
            "style_preset": "photographic",
        }
    },

    "maybe-derivative-post": {
        "post": lambda settings: [] if coin_flip() else [Preset("derivative-post")]
    },

    "maybe-invert": {
        "post": lambda settings: [] if coin_flip() else [Preset("invert")]
    },

    "maybe-palette": {
        "settings": lambda: {
            "palette_alpha": 0.5 + random.random() * 0.5,
            "palette_name": random_member(PALETTES),
            "palette_on": random.random() < 0.375,
        },
        "post": lambda settings: [] if not settings["palette_on"] else [
            Effect("palette", name=settings["palette_name"], alpha=settings["palette_alpha"])
        ]
    },

    "maybe-rotate": {
        "settings": lambda: {
            "angle": random.random() * 360.0
        },
        "post": lambda settings: [] if coin_flip() else [Effect("rotate", angle=settings["angle"])]
    },

    "maybe-skew": {
        "final": lambda settings: [] if coin_flip() else [Preset("skew")]
    },

    "mcpaint": {
        "layers": ["glyph-map", "skew", "grain", "vignette-dark", "brightness-final", "contrast-final", "saturation"],
        "settings": lambda: {
            "corners": True,
            "freq": random.randint(2, 8),
            "glyph_map_colorize": False,
            "glyph_map_mask": mask.mcpaint,
            "glyph_map_zoom": random.randint(2, 4),
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "macpaint, classic computing, truchet pattern, bitmap, 1-bit graphics",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "pixel-art",
        }
    },

    "moire-than-a-feeling": {
        "layers": ["basic", "wormhole", "density-map", "invert", "contrast-post"],
        "settings": lambda: {
            "octaves": random.randint(1, 2),
            "saturation": 0,
            "wormhole_kink": 128,
            "wormhole_stride": 0.0005,
        },
        "ai": {
            "prompt": "interference patterns, moire",
            "image_strength": 0.875,
        }
    },

    "molten-glass": {
        "layers": ["basic", "sine-octaves", "octave-warp-post", "brightness-post", "contrast-post",
                   "bloom", "shadow", "normalize", "lens"],
        "settings": lambda: {
            "hue_range": random.random() * 3.0,
        },
        "ai": {
            "prompt": "molten glass, glass blowing art, melted paint, melting colors, unmixed",
            "image_strength": 0.425,
            "cfg_scale": 25,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "multires": {
        "layers": ["basic"],
        "settings": lambda: {
            "octaves": random.randint(4, 8)
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "tile-texture",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "multires-alpha": {
        "layers": ["multires"],
        "settings": lambda: {
            "distrib": distrib.exp,
            "lattice_drift": 1,
            "octave_blending": blend.alpha,
            "octaves": 5,
            "palette_on": False,
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "multires-low": {
        "layers": ["basic"],
        "settings": lambda: {
            "octaves": random.randint(2, 4)
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "multires-ridged": {
        "layers": ["multires"],
        "settings": lambda: {
            "lattice_drift": random.random(),
            "ridges": True
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "muppet-fur": {
        "layers": ["basic", "worms", "rotate", "bloom", "lens"],
        "settings": lambda: {
            "hue_range": random.random() * 0.5,
            "hue_rotation": random.random(),
            "lattice_drift": 1.0,
            "palette_on": False,
            "worms_alpha": 0.625 + random.random() * 0.125,
            "worms_behavior": worms.unruly if coin_flip() else worms.obedient,
            "worms_density": 250,
            "worms_stride": 0.75,
            "worms_stride_deviation": 0.25,
        },
        "ai": {
            "prompt": "flow field, colorful faux fur, furry, fuzzy, fluffy, rave culture",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "mycelium": {
        "layers": ["multires", "grayscale", "octave-warp-octaves", "derivative-post",
                   "normalize", "fractal-seed", "vignette-dark", "contrast-post"],
        "settings": lambda: {
            "color_space": color.grayscale,
            "distrib": distrib.ones,
            "freq": [random.randint(3, 4), random.randint(3, 4)],
            "lattice_drift": 1.0,
            "mask": mask.h_tri,
            "mask_static": True,
            "speed": 0.05,
            "warp_freq": [random.randint(2, 3), random.randint(2, 3)],
            "warp_range": 2.5 + random.random() * 1.25,
            "worms_behavior": worms.random,
        },
        "ai": {
            "prompt": "flow field, mycelium, roots, mycelial network",
            "image_strength": 0.75,
            "cfg_scale": 30,
            "style_preset": "photographic",
        }
    },

    "nausea": {
        "layers": ["value-mask", "ripple", "normalize", "aberration"],
        "settings": lambda: {
            "color_space": color.rgb,
            "mask": random_member([mask.h_bar, mask.v_bar]),
            "mask_repeat": random.randint(5, 8),
            "ripple_kink": 1.25 + random.random() * 1.25,
            "ripple_freq": random.randint(2, 3),
            "ripple_range": 1.25 + random.random(),
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "groovy, psychedelia, peace and love, vintage",
            "image_strength": 0.95,
            "cfg_scale": 25,
        }
    },

    "nebula": {
        "final": lambda settings: [Effect("nebula")]
    },

    "nerdvana": {
        "layers": ["symmetry", "voronoi", "density-map", "reverb", "bloom"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "palette_on": False,
            "reverb_octaves": 2,
            "reverb_ridges": False,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": random.randint(5, 10),
            "voronoi_nth": 1,
        },
        "ai": {
            "prompt": "trippy kaleidoscopic imagery, symmetry, mandala, nirvana",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "neon-cambrian": {
        "layers": ["voronoi", "posterize", "wormhole", "derivative-post", "brightness-final", "bloom", "contrast-final", "aberration"],
        "settings": lambda: {
            "contrast_final": 4.0,
            "dist_metric": distance.euclidean,
            "freq": 12,
            "hue_range": 4,
            "posterize_levels": random.randint(20, 25),
            "voronoi_diagram_type": voronoi.color_flow,
            "voronoi_point_distrib": point.random,
            "wormhole_stride": 0.2 + random.random() * 0.1,
        },
        "ai": {
            "prompt": "artistic depiction of primordial underwater life, cambrian era, kelp, invertebrates",
            "image_strength": 0.875,
            "cfg_scale": 20,
        }
    },

    "noise-blaster": {
        "layers": ["multires", "reindex-octaves", "reindex-post", "grain"],
        "settings": lambda: {
            "freq": random.randint(3, 4),
            "lattice_drift": 1,
            "reindex_range": 3,
            "speed": 0.025,
        },
        "ai": {
            "prompt": "psychedelic fractal turbulence",
            "image_strength": 0.25,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        },
    },

    "noise-lake": {
        "layers": ["multires-low", "value-refract", "snow", "lens"],
        "settings": lambda: {
            "hue_range": 0.75 + random.random() * 0.375,
            "freq": random.randint(4, 6),
            "lattice_drift": 1.0,
            "ridges": True,
            "value_freq": random.randint(4, 6),
            "value_refract_range": 0.25 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "psychedelic fractal vibrations, lake ripples",
            "image_strength": 0.75,
            "cfg_scale": 20,
        },
    },

    "noise-tunnel": {
        "layers": ["basic", "periodic-distance", "periodic-refract", "lens"],
        "settings": lambda: {
            "hue_range": 2.0 + random.random(),
            "speed": 1.0,
        },
        "ai": {
            "prompt": "tunnel of refracting light",
            "image_strength": 0.5,
            "cfg_scale": 30,
        },
    },

    "noirmaker": {
        "layers": ["grain", "grayscale", "light-leak", "bloom", "contrast-final", "vignette-dark"],
    },

    "normals": {
        "final": lambda settings: [Effect("normal_map")]
    },

    "normalize": {
        "post": lambda settings: [Effect("normalize")]
    },

    "now": {
        "layers": ["multires-low", "normalize", "wobble", "voronoi", "funhouse", "outline", "grain", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "freq": random.randint(3, 10),
            "hue_range": random.random(),
            "lattice_drift": coin_flip(),
            "saturation": 0.5 + random.random() * 0.5,
            "spline_order": interp.constant,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(3, 10),
            "voronoi_refract": 2.0 + random.random(),
            "warp_freq": random.randint(2, 4),
            "warp_octaves": 1,
            "warp_range": 0.0375 + random.random() * 0.0375,
            "warp_spline_order": interp.bicubic,
        },
        "ai": {
            "prompt": "now, psychedelic fractal imagery",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "nudge-hue": {
        "final": lambda settings: [Effect("adjust_hue", amount=-.125)]
    },

    "numberwang": {
        "layers": ["value-mask", "funhouse", "posterize", "palette", "maybe-invert",
                   "random-hue", "grain", "saturation"],
        "settings": lambda: {
            "mask": mask.alphanum_numeric,
            "mask_repeat": random.randint(5, 10),
            "posterize_levels": 2,
            "spline_order": interp.cosine,
            "warp_range": 0.25 + random.random() * 0.75,
            "warp_freq": random.randint(2, 4),
            "warp_octaves": 1,
            "warp_spline_order": interp.bicubic,
        },
        "ai": {
            "prompt": "distorted numbers and symbols, glyphs",
        }
    },

    "octave-blend": {
        "layers": ["multires-alpha"],
        "settings": lambda: {
            "corners": True,
            "distrib": random_member([distrib.ones, distrib.uniform]),
            "freq": random.randint(2, 5),
            "lattice_drift": 0,
            "mask": random_member(mask.procedural_members()),
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "colorful overlapping squares",
            "image_strength": 0.5,
            "cfg_scale": 30,
            "style_preset": "pixel-art",
        }
    },

    "octave-warp-octaves": {
        "settings": lambda: {
            "warp_freq": [random.randint(2, 4), random.randint(2, 4)],
            "warp_octaves": random.randint(1, 4),
            "warp_range": 0.5 + random.random() * 0.25,
            "warp_signed_range": False,
            "warp_spline_order": interp.bicubic
        },
        "octaves": lambda settings: [
            Effect("warp",
                   displacement=settings["warp_range"],
                   freq=settings["warp_freq"],
                   octaves=settings["warp_octaves"],
                   signed_range=settings["warp_signed_range"],
                   spline_order=settings["warp_spline_order"])
        ]
    },

    "octave-warp-post": {
        "settings": lambda: {
            "speed": 0.025 + random.random() * 0.0125,
            "warp_freq": random.randint(2, 3),
            "warp_octaves": random.randint(2, 4),
            "warp_range": 2.0 + random.random(),
            "warp_spline_order": interp.bicubic,
        },
        "post": lambda settings: [
            Effect("warp",
                   displacement=settings["warp_range"],
                   freq=settings["warp_freq"],
                   octaves=settings["warp_octaves"],
                   spline_order=settings["warp_spline_order"])
        ]
    },

    "oldschool": {
        "layers": ["voronoi", "normalize", "random-hue", "saturation", "distressed"],
        "settings": lambda: {
            "color_space": color.rgb,
            "corners": True,
            "dist_metric": distance.euclidean,
            "distrib": distrib.ones,
            "freq": random.randint(2, 5) * 2,
            "mask": mask.chess,
            "spline_order": interp.constant,
            "speed": 0.05,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_distrib": point.random,
            "voronoi_point_freq": random.randint(4, 8),
            "voronoi_refract": random.randint(8, 12) * 0.5,
        },
        "ai": {
            "prompt": "a distorted psychedelic black and white checker pattern, groovy, psychedelia, peace and love, vintage, monochrome",
            "image_strength": 0.625,
            "cfg_scale": 30,
        }
    },

    "one-art-please": {
        "layers": ["contrast-post", "grain", "light-leak", "saturation", "texture"],
    },

    "oracle": {
        "layers": ["value-mask", "random-hue", "maybe-invert", "crt"],
        "settings": lambda: {
            "corners": True,
            "mask": mask.iching,
            "mask_repeat": random.randint(1, 8),
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "oracle, i-ching hexagram"
        }
    },

    "outer-limits": {
        "layers": ["symmetry", "reindex-post", "normalize", "grain", "be-kind-rewind", "vignette-dark", "contrast-post"],
        "settings": lambda: {
            "palette_on": False,
            "reindex_range": random.randint(8, 16),
            "saturation": 0,
        },
        "ai": {
            "prompt": "spooky concentric rings, the outer limits intro, retro tv",
        }
    },

    "outline": {
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "outline_invert": False,
        },
        "post": lambda settings: [
            Effect("outline",
                sobel_metric=settings["dist_metric"],
                invert=settings["outline_invert"],
            )
        ]
    },

    "oxidize": {
        "layers": ["multires", "refract-post", "contrast-post", "bloom", "shadow", "saturation", "lens"],
        "settings": lambda: {
            "distrib": distrib.exp,
            "freq": 4,
            "hue_range": 0.875 + random.random() * 0.25,
            "lattice_drift": 1,
            "octave_blending": blend.reduce_max,
            "octaves": 8,
            "refract_range": 0.1 + random.random() * 0.05,
            "saturation_final": 0.5,
            "speed": 0.05,
        },
        "ai": {
            "prompt": "oxidation, oxidized metal, rusted iron, corrosive, corrosion, sulfurous, rough texture",
            "image_strength": 0.625,
            "cfg_scale": 30,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "paintball-party": {
        "layers": ["basic"] + ["spatter-post"] * random.randint(1, 4) +
                  ["spatter-final"] * random.randint(1, 4) + ["bloom"],
        "settings": lambda: {
            "distrib": random_member([distrib.zeros, distrib.ones]),
        },
        "ai": {
            "prompt": "splattered paint, dripping paint, splat, splash, splatter, spatter"
        }
    },

    "painterly": {
        "layers": ["value-mask", "ripple", "funhouse", "maybe-rotate", "saturation", "grain"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "hue_range": 0.333 + random.random() * 0.666,
            "mask": random_member(mask.grid_members()),
            "mask_repeat": 1,
            "octaves": 8,
            "ridges": True,
            "ripple_freq": random.randint(4, 6),
            "ripple_kink": 0.0625 + random.random() * 0.125,
            "ripple_range": 0.0625 + random.random() * 0.125,
            "spline_order": interp.linear,
            "warp_freq": random.randint(5, 7),
            "warp_octaves": 8,
            "warp_range": 0.0625 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "abstract watercolor, fine art, painterly, textured canvas",
            "image_strength": 0.375,
            "cfg_scale": 30,
        }
    },

    "palette": {
        "layers": ["maybe-palette"],
        "settings": lambda: {
            "palette_name": random_member(PALETTES),
            "palette_on": True,
        },
    },

    "pantheon": {
        "layers": ["runes-of-arecibo"],
        "settings": lambda: {
            "mask": random_member([mask.invaders_square, random_member(mask.glyph_members())]),
            "mask_repeat": random.randint(2, 3) * 2,
            "octaves": 2,
            "posterize_levels": random.randint(3, 6),
            "refract_range": random_member([0, random.random() * 0.05]),
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "ancient mayan gods, dieties, stone face carving, stone tablet, pantheon of mayan gods, ancient maya art and language",
            "image_strength": 0.625,
            "cfg_scale": 25,
            "style_preset": "photographic",
        },
    },

    "pearlescent": {
        "layers": ["voronoi", "normalize", "refract-post", "brightness-final", "bloom", "shadow", "lens"],
        "settings": lambda: {
            "brightness_final": 0.05,
            "dist_metric": distance.euclidean,
            "freq": [2, 2],
            "hue_range": random.randint(3, 5),
            "octaves": random.randint(3, 5),
            "refract_range": 0.5 + random.random() * 0.25,
            "ridges": coin_flip(),
            "saturation": 0.175 + random.random() * 0.25,
            "tint_alpha": 0.0125 + random.random() * 0.0625,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_refract": 0.25 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "iridescent material, iridescence, mica, silicate mineral, mother-of-pearl",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "periodic-distance": {
        "layers": ["basic"],
        "settings": lambda: {
            "freq": random.randint(1, 6),
            "distrib": random_member([m for m in distrib if distrib.is_center_distance(m)]),
            "hue_range": 0.25 + random.random() * 0.125,
        },
        "post": lambda settings: [Effect("normalize")],
        "ai": {
            "prompt": "abstract psychedelic art, concentric shapes",
            "image_strength": 0.25,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "periodic-refract": {
        "layers": ["value-refract"],
        "settings": lambda: {
            "value_distrib": random_member([m for m in distrib if distrib.is_center_distance(m) or distrib.is_scan(m)]),
        },
    },

    "pink-diamond": {
        "layers": ["periodic-distance", "periodic-refract", "refract-octaves", "refract-post", "nudge-hue", "bloom", "lens"],
        "settings": lambda: {
            "color_space": color.hsv,
            "bloom_alpha": 0.333 + random.random() * 0.16667,
            "brightness_distrib": distrib.uniform,
            "freq": 2,
            "hue_range": 0.2 + random.random() * 0.1,
            "hue_rotation": 0.9 + random.random() * 0.05,
            "palette_on": False,
            "refract_range": 0.0125 + random.random() * 0.00625,
            "refract_y_from_offset": False,
            "ridges": True,
            "saturation_distrib": distrib.ones,
            "speed": -0.125,
            "value_distrib": random_member([m for m in distrib if distrib.is_center_distance(m)]),
            "vaseline_alpha": 0.125 + random.random() * 0.0625,
        },
        "generator": lambda settings: {
            "distrib": settings["value_distrib"],
        },
        "ai": {
            "prompt": "light refracted through a pink gemstone with orange highlights",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "pixel-sort": {
        "settings": lambda: {
            "pixel_sort_angled": coin_flip(),
            "pixel_sort_darkest": coin_flip(),
        },
        "final": lambda settings: [
            Effect("pixel_sort",
                   angled=settings["pixel_sort_angled"],
                   darkest=settings["pixel_sort_darkest"])
        ]
    },

    "plaid": {
        "layers": ["multires-low", "derivative-octaves", "funhouse", "maybe-rotate", "grain",
                   "vignette-dark", "saturation"],
        "settings": lambda: {
            "dist_metric": distance.chebyshev,
            "distrib": distrib.ones,
            "freq": random.randint(2, 4) * 2,
            "hue_range": random.random() * 0.5,
            "mask": mask.chess,
            "spline_order": random.randint(1, 3),
            "vignette_alpha": 0.25 + random.random() * 0.125,
            "warp_freq": random.randint(2, 3),
            "warp_range": random.random() * 0.125,
            "warp_octaves": 1,
        },
        "ai": {
            "prompt": "plaid fabric, flannel, tartan, soft cotton fabric",
            "style_preset": "photographic",
        }
    },

    "pluto": {
        "layers": ["multires-ridged", "derivative-octaves", "voronoi", "refract-post",
                   "bloom", "shadow", "contrast-post", "grain", "saturation", "lens"],
        "settings": lambda: {
            "deriv_alpha": 0.333 + random.random() * 0.16667,
            "dist_metric": distance.euclidean,
            "distrib": distrib.exp,
            "freq": random.randint(4, 8),
            "hue_rotation": 0.575,
            "octave_blending": blend.reduce_max,
            "palette_on": False,
            "refract_range": 0.01 + random.random() * 0.005,
            "saturation": 0.75 + random.random() * 0.25,
            "shadow_alpha": 1.0,
            "tint_alpha": 0.0125 + random.random() * 0.00625,
            "vignette_alpha": 0.125 + random.random() * 0.0625,
            "voronoi_alpha": 0.925 + random.random() * 0.075,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 2,
            "voronoi_point_distrib": point.random,
        },
        "ai": {
            "prompt": "pluto's bladed terrain, geomorphology, nasa imagery, new horizons, terrain map, high-relief, ice, frozen",
            "image_strength": 0.3,
            "cfg_scale": 30,
            "style_preset": "photographic",
        }
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

    "precision-error": {
        "layers": ["symmetry", "derivative-octaves", "reflect-octaves", "derivative-post",
                   "density-map", "invert", "shadows", "contrast-post"],
        "settings": lambda: {
            "palette_on": False,
            "reflect_range": 0.75 + random.random() * 2.0,
        },
        "ai": {
            "prompt": "an abstract artistic representation of floating point error",
        }
    },

    "procedural-mask": {
        "layers": ["value-mask", "skew", "bloom", "crt", "vignette-dark", "contrast-final"],
        "settings": lambda: {
            "spline_order": interp.cosine,
            "mask": random_member(mask.procedural_members()),
            "mask_repeat": random.randint(10, 20)
        },
        "ai": {
            "prompt": "stylized distorted symbols, glyphs, truchet pattern",
        }
    },

    "prophesy": {
        "layers": ["value-mask", "refract-octaves", "posterize", "emboss", "maybe-invert",
                   "tint", "shadows", "saturation", "dexter", "texture", "maybe-skew", "grain"],
        "settings": lambda: {
            "grain_brightness": 0.125,
            "grain_contrast": 1.125,
            "mask": random_member(mask.glyph_members()),
            "mask_repeat": random.randint(3, 7),
            "octaves": 2,
            "palette_on": False,
            "posterize_levels": random.randint(3, 6),
            "saturation": 0.25 + random.random() * 0.125,
            "spline_order": interp.cosine,
            "refract_range": 0.0125 + random.random() * 0.025,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "tint_alpha": 0.01 + random.random() * 0.005,
            "vignette_alpha": 0.25 + random.random() * 0.125,
        },
        "ai": {
            "prompt": "mayan glyph writing, codex, ancient mayan stone carving, maya stelae, mayan art and language, stone tablet, a grid of entities, pantheon of mayan gods, psychedelic visionary art, ancient language",
            "image_strength": 0.625,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "pull": {
        "layers": ["basic-voronoi", "erosion-worms"],
        "settings": lambda: {
            "voronoi_alpha": 0.25 + random.random() * 0.5,
            "voronoi_diagram_type": random_member([voronoi.range, voronoi.color_range, voronoi.range_regions]),
        },
        "ai": {
            "prompt": "flow field, abstract psychedelic fractal pattern",
            "image_strength": 0.375,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "puzzler": {
        "layers": ["basic-voronoi", "maybe-invert", "wormhole"],
        "settings": lambda: {
            "speed": 0.025,
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_point_distrib": random_member(point, mask.nonprocedural_members()),
            "voronoi_point_freq": 10,
        },
        "ai": {
            "prompt": "cut out pieces of paper",
            "cfg_scale": 25,
            "style_preset": "origami",
        }
    },

    "quadrants": {
        "layers": ["basic", "reindex-post"],
        "settings": lambda: {
            "color_space": color.rgb,
            "freq": [2, 2],
            "reindex_range": 2,
            "spline_order": random_member([interp.cosine, interp.bicubic]),
        },
        "ai": {
            "prompt": "abstract fractal design",
            "image_strength": 0.375,
            "cfg_scale": 25,
        }
    },

    "quilty": {
        "layers": ["voronoi", "skew", "bloom", "grain"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.chebyshev]),
            "freq": random.randint(2, 4),
            "saturation": random.random() * 0.5,
            "spline_order": interp.constant,
            "voronoi_diagram_type": random_member([voronoi.range, voronoi.color_range]),
            "voronoi_nth": random.randint(0, 4),
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": random.randint(2, 4),
            "voronoi_refract": random.randint(1, 3) * 0.5,
            "voronoi_refract_y_from_offset": True,
        },
        "ai": {
            "prompt": "patchwork quilt, soft cotton fabric",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "random-hue": {
        "final": lambda settings: [Effect("adjust_hue", amount=random.random())]
    },

    "rasteroids": {
        "layers": ["basic", "funhouse", "sobel", "invert", "pixel-sort", "bloom", "crt", "vignette-dark"],
        "settings": lambda: {
            "distrib": random_member([distrib.uniform, distrib.ones]),
            "freq": 6 * random.randint(2, 3),
            "mask": random_member(mask),
            "pixel_sort_angled": False,
            "pixel_sort_darkest": False,
            "spline_order": interp.constant,
            "vignette_alpha": 0.125 + random.random() * 0.0625,
            "warp_freq": random.randint(3, 5),
            "warp_octaves": random.randint(3, 5),
            "warp_range": 0.125 + random.random() * 0.0625,
            "warp_spline_order": interp.constant,
        },
        "ai": {
            "prompt": "vector display reminiscent of classic arcade games, neon squares, rasterized shapes, glowing outlines on a black background",
        }
    },

    "reflect-octaves": {
        "settings": lambda: {
            "reflect_range": 5 + random.random() * 0.25,
        },
        "octaves": lambda settings: [
            Effect("refract",
                   displacement=settings["reflect_range"],
                   from_derivative=True)
        ]
    },

    "reflect-post": {
        "settings": lambda: {
            "reflect_range": 0.5 + random.random() * 12.5,
        },
        "post": lambda settings: [
            Effect("refract",
                   displacement=settings["reflect_range"],
                   from_derivative=True)
        ]
    },

    "reflecto": {
        "layers": ["basic", "reflect-octaves", "reflect-post", "grain"],
        "ai": {
            "prompt": "distorted funhouse mirror reflecting a psychedelic fractal pattern",
            "image_strength": 0.5,
            "cfg_scale": 30,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "refract-octaves": {
        "settings": lambda: {
            "refract_range": 0.5 + random.random() * 0.25,
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
            "refract_range": 0.125 + random.random() * 1.25,
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

    "regional": {
        "layers": ["voronoi", "glyph-map", "bloom", "crt", "contrast-post"],
        "settings": lambda: {
            "glyph_map_colorize": coin_flip(),
            "glyph_map_zoom": random.randint(4, 8),
            "hue_range": 0.25 + random.random(),
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_nth": 0,
        },
        "ai": {
            "prompt": "a grid of glyphs and symbols, truchet pattern",
            "image_strength": 0.5,
            "cfg_scale": 30,
        }
    },

    "reindex-octaves": {
        "settings": lambda: {
            "reindex_range": 0.125 + random.random() * 2.5
        },
        "octaves": lambda settings: [Effect("reindex", displacement=settings["reindex_range"])]
    },

    "reindex-post": {
        "settings": lambda: {
            "reindex_range": 0.125 + random.random() * 2.5
        },
        "post": lambda settings: [Effect("reindex", displacement=settings["reindex_range"])]
    },

    "remember-logo": {
        "layers": ["symmetry", "voronoi", "derivative-post", "density-map", "crt", "vignette-dark"],
        "settings": lambda: {
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.regions,
            "voronoi_nth": random.randint(0, 4),
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": random.randint(3, 7),
        },
        "ai": {
            "prompt": "retro vector design",
            "image_strength": 0.5,
            "cfg_scale": 30,
        }
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

    "ride-the-rainbow": {
        "layers": ["basic", "swerve-v", "scuff", "distressed", "contrast-post"],
        "settings": lambda: {
            "brightness_distrib": distrib.ones,
            "corners": True,
            "distrib": distrib.column_index,
            "freq": random.randint(6, 12),
            "hue_range": 0.9,
            "palette_on": False,
            "saturation_distrib": distrib.ones,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "abstract design, art, vintage psychedelia, rainbow, pride",
            "image_strength": 0.75,
            "cfg_scale": 25,
        }
    },

    "ridge": {
        "post": lambda settings: [Effect("ridge")]
    },

    "ripple": {
        "settings": lambda: {
            "ripple_range": 0.025 + random.random() * 0.1,
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

    "runes-of-arecibo": {
        "layers": ["value-mask", "refract-octaves", "posterize", "emboss", "maybe-invert", "contrast-post", "skew",
                   "grain", "texture", "vaseline", "brightness-final", "contrast-final"],
        "settings": lambda: {
            "brightness_final": -0.1,
            "color_space": color.grayscale,
            "corners": True,
            "mask": random_member([mask.arecibo_num, mask.arecibo_bignum, mask.arecibo_nucleotide]),
            "mask_repeat": random.randint(4, 12),
            "palette_on": False,
            "posterize_levels": random.randint(1, 3),
            "refract_range": 0.025 + random.random() * 0.0125,
            "refract_signed_range": False,
            "refract_y_from_offset": True,
            "spline_order": random_member([interp.linear, interp.cosine]),
        },
        "ai": {
            "prompt": "alien glyph writing, codex, ancient alien stone carving, alien language, stone tablet",
            "image_strength": 0.625,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "sands-of-time": {
        "layers": ["basic", "worms", "lens"],
        "settings": lambda: {
            "freq": random.randint(3, 5),
            "octaves": random.randint(1, 3),
            "worms_behavior": worms.unruly,
            "worms_alpha": 1,
            "worms_density": 750,
            "worms_duration": 0.25,
            "worms_kink": random.randint(1, 2),
            "worms_stride": random.randint(128, 256),
        },
        "ai": {
            "prompt": "sand blowing away in the wind, sand scattered to the wind, particles",
            "image_strength": 0.75,
            "cfg_scale": 20,
            "style_preset": "photographic",
        }
    },

    "satori": {
        "layers": ["multires-low", "sine-octaves", "voronoi", "contrast-post", "grain", "saturation"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "dist_metric": random_member(distance.absolute_members()),
            "freq": random.randint(3, 4),
            "hue_range": random.random(),
            "lattice_drift": 1,
            "ridges": True,
            "speed": 0.05,
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_refract": random.randint(6, 12) * 0.25,
            "voronoi_point_distrib": random_member([point.random] + point.circular_members()),
            "voronoi_point_freq": random.randint(2, 8),
        },
        "ai": {
            "prompt": "abstract psychedelic fractal pattern",
            "image_strength": 0.25,
            "cfg_scale": 20,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "saturation": {
        "settings": lambda: {
            "saturation_final": 0.333 + random.random() * 0.16667
        },
        "final": lambda settings: [Effect("adjust_saturation", amount=settings["saturation_final"])]
    },

    "sblorp": {
        "layers": ["basic", "posterize", "invert", "grain", "saturation"],
        "settings": lambda: {
            "color_space": color.rgb,
            "distrib": distrib.ones,
            "freq": random.randint(5, 9),
            "lattice_drift": 1.25 + random.random() * 1.25,
            "mask": mask.sparse,
            "octave_blending": blend.reduce_max,
            "octaves": random.randint(2, 3),
            "posterize_levels": 1,
        },
        "ai": {
            "prompt": "psychedelic fractal imagery, high-contrast ooze splattered onto a bright background",
        }
    },

    "sbup": {
        "layers": ["basic", "posterize", "funhouse", "falsetto", "palette", "distressed"],
        "settings": lambda: {
            "distrib": distrib.ones,
            "freq": [2, 2],
            "mask": mask.square,
            "posterize_levels": random.randint(1, 2),
            "warp_range": 1.5 + random.random(),
        },
        "ai": {
            "prompt": "psychedelic fractal imagery, noisy pattern on a bright background",
        }
    },

    "scanline-error": {
        "final": lambda settings: [Effect("scanline_error")]
    },

    "scratches": {
        "final": lambda settings: [Effect("scratches")]
    },

    "scribbles": {
        "layers": ["basic", "derivative-octaves", "derivative-post", "derivative-post", "contrast-post", "invert", "sketch"],
        "settings": lambda: {
            "color_space": color.grayscale,
            "deriv_alpha": 0.925,
            "freq": random.randint(2, 4),
            "lattice_drift": 1.0,
            "octaves": random.randint(3, 4),
            "palette_on": False,
            "ridges": True,
        },
        "ai": {
            "prompt": "scribbles, doodles, style of pencil drawing, sketch",
            "image_strength": 0.75,
            "cfg_scale": 25,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "scuff": {
        "final": lambda settings: [Effect("scratches")]
    },

    "serene": {
        "layers": ["basic-water", "periodic-refract", "refract-post", "lens"],
        "settings": lambda: {
            "freq": random.randint(2, 3),
            "octaves": 3,
            "refract_range": 0.0025 + random.random() * 0.00125,
            "refract_y_from_offset": False,
            "value_distrib": distrib.center_circle,
            "value_freq": random.randint(2, 3),
            "value_refract_range": 0.025 + random.random() * 0.0125,
            "speed": 0.25,
        },
        "ai": {
            "prompt": "outward ripples, serene, peaceful",
            "image_strength": 0.5,
            "cfg_scale": 20,
            "style_preset": "photographic",
        }
    },

    "shadow": {
        "settings": lambda: {
            "shadow_alpha": 0.5 + random.random() * 0.25
        },
        "post": lambda settings: [Effect("shadow", alpha=settings["shadow_alpha"])]
    },

    "shadows": {
        "layers": ["shadow", "vignette-dark"]
    },

    "shake-it-like": {
        "post": lambda settings: [Effect("frame")]
    },

    "shape-party": {
        "layers": ["voronoi", "posterize", "invert", "aberration", "grain", "saturation"],
        "settings": lambda: {
            "aberration_displacement": 0.125 + random.random() * 0.0625,
            "color_space": color.rgb,
            "dist_metric": distance.manhattan,
            "distrib": distrib.ones,
            "freq": 11,
            "mask": random_member(mask.procedural_members()),
            "posterize_levels": 1,
            "spline_order": interp.cosine,
            "voronoi_point_freq": 2,
            "voronoi_nth": 1,
            "voronoi_refract": 0.125 + random.random() * 0.25,
        },
        "ai": {
            "prompt": "abstract modern art, the shapes are having a party, festive shapes, distorted glyphs",
        }
    },

    "shatter": {
        "layers": ["basic-voronoi", "refract-post", "posterize-outline", "maybe-invert", "normalize", "lens", "grain"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "dist_metric": random_member(distance.absolute_members()),
            "posterize_levels": random.randint(4, 6),
            "refract_range": 0.75 + random.random() * 0.375,
            "refract_y_from_offset": True,
            "speed": 0.05,
            "voronoi_inverse": coin_flip(),
            "voronoi_point_freq": random.randint(3, 5),
            "voronoi_diagram_type": voronoi.range_regions,
        },
        "ai": {
            "prompt": "shattered shapes, broken, shards, fragments",
            "image_strength": 0.375,
            "cfg_scale": 25,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "shimmer": {
        "layers": ["basic", "derivative-octaves", "voronoi", "refract-post", "lens"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "freq": random.randint(2, 3),
            "hue_range": 3.0 + random.random() * 1.5,
            "lattice_drift": 1.0,
            "refract_range": 1.25 * random.random() * 0.625,
            "ridges": True,
            "voronoi_alpha": 0.25 + random.random() * 0.125,
            "voronoi_diagram_type": voronoi.color_flow,
            "voronoi_point_freq": 10,
        },
        "ai": {
            "prompt": "psychedelic fractal imagery, shimmering noise",
            "image_strength": 0.375,
            "cfg_scale": 25,
        }
    },

    "shmoo": {
        "layers": ["basic", "posterize", "invert", "outline", "distressed"],
        "settings": lambda: {
            "freq": random.randint(3, 4),
            "hue_range": 1.5 + random.random() * 0.75,
            "palette_on": False,
            "posterize_levels": random.randint(1, 4),
            "speed": 0.025,
        },
        "ai": {
            "prompt": "1970s cartoon blobs, colorful, outlined regions of solid color",
        }
    },

    "sideways": {
        "layers": ["multires-low", "reflect-octaves", "pixel-sort", "lens", "crt"],
        "settings": lambda: {
            "freq": random.randint(6, 12),
            "distrib": distrib.ones,
            "mask": mask.script,
            "palette_on": False,
            "pixel_sort_angled": False,
            "saturation": 0.0625 + random.random() * 0.125,
            "spline_order": random_member([m for m in interp if m != interp.constant]),
        },
        "ai": {
            "prompt": "psychedelic fractal imagery, sideways",
        }
    },

    "simple-frame": {
        "post": lambda settings: [Effect("simple_frame")]
    },

    "sined-multifractal": {
        "layers": ["multires-ridged", "sine-octaves", "grain", "saturation"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "freq": random.randint(2, 3),
            "hue_range": random.random(),
            "hue_rotation": random.random(),
            "lattice_drift": 0.75,
            "palette_on": False,
            "sine_range": random.randint(10, 15),
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
            "image_strength": 0.375,
            "cfg_scale": 30,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "sine-octaves": {
        "settings": lambda: {
            "sine_range": random.randint(8, 12),
            "sine_rgb": False,
        },
        "octaves": lambda settings: [
            Effect("sine", amount=settings["sine_range"], rgb=settings["sine_rgb"])
        ]
    },

    "sine-post": {
        "settings": lambda: {
            "sine_range": random.randint(8, 20),
            "sine_rgb": True,
        },
        "post": lambda settings: [
            Effect("sine", amount=settings["sine_range"], rgb=settings["sine_rgb"])
        ]
    },

    "singularity": {
        "layers": ["basic-voronoi", "grain"],
        "settings": lambda: {
            "voronoi_point_freq": 1,
            "voronoi_diagram_type": random_member([voronoi.color_range, voronoi.range, voronoi.range_regions]),
        },
        "ai": {
            "prompt": "abstract psychedelic imagery, singularity",
        }
    },

    "sketch": {
        "layers": ["fibers", "grime", "texture"],
        "post": lambda settings: [Effect("sketch")],
    },

    "skew": {
        "layers": ["rotate"],
        "settings": lambda: {
            "angle": random.randint(-10, 10),
        },
    },

    "snow": {
        "settings": lambda: {
            "snow_alpha": 0.125 + random.random() * 0.0625
        },
        "final": lambda settings: [Effect("snow", alpha=settings["snow_alpha"])]
    },

    "sobel": {
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
        },
        "post": lambda settings: [Effect("sobel", dist_metric=settings["dist_metric"])]
    },

    "soft-cells": {
        "layers": ["voronoi", "maybe-rotate", "lens", "bloom"],
        "settings": lambda: {
            "color_space": random_member(color.color_members()),
            "freq": 2,
            "hue_range": 0.25 + random.random() * 0.25,
            "hue_rotation": random.random(),
            "lattice_drift": 1,
            "octaves": random.randint(1, 4),
            "voronoi_alpha": 0.5 + random.random() * 0.5,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_point_distrib": random_member(point, mask.nonprocedural_members()),
            "voronoi_point_freq": random.randint(4, 8),
        },
        "ai": {
            "prompt": "softly glowing shapes, soft cells",
        }
    },

    "soup": {
        "layers": ["voronoi", "normalize", "refract-post", "worms",
                   "grayscale", "density-map", "bloom", "shadow", "lens"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "freq": random.randint(2, 3),
            "refract_range": 2.5 + random.random() * 1.25,
            "refract_y_from_offset": True,
            "speed": 0.025,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_inverse": True,
            "voronoi_point_freq": random.randint(2, 3),
            "worms_alpha": 0.75 + random.random() * 0.25,
            "worms_behavior": worms.random,
            "worms_density": 500,
            "worms_kink": 4.0 + random.random() * 2.0,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.0,
        },
        "ai": {
            "prompt": "fractal flame, psychedelic fractal imagery, flow field",
        }
    },

    "spaghettification": {
        "layers": ["multires-low", "voronoi", "worms", "funhouse", "contrast-post", "density-map", "lens"],
        "settings": lambda: {
            "freq": 2,
            "palette_on": False,
            "voronoi_diagram_type": voronoi.flow,
            "voronoi_inverse": True,
            "voronoi_point_freq": 1,
            "warp_range": 0.5 + random.random() * 0.25,
            "warp_octaves": 1,
            "worms_alpha": 0.875,
            "worms_behavior": worms.chaotic,
            "worms_density": 1000,
            "worms_kink": 1.0,
            "worms_stride": random.randint(150, 250),
            "worms_stride_deviation": 0.0,
        },
        "ai": {
            "prompt": "fractal flame, psychedelic fractal imagery, flow field",
        }
    },

    "spectrogram": {
        "layers": ["basic", "grain", "filthy"],
        "settings": lambda: {
            "distrib": distrib.row_index,
            "freq": random.randint(256, 512),
            "hue_range": 0.5 + random.random() * 0.5,
            "mask": mask.bar_code,
            "spline_order": interp.constant,
        },
        "ai": {
            "prompt": "stellar spectra, spectrum, spectrogram, spectrographic",
        }
    },

    "spatter-post": {
        "settings": lambda: {
            "speed": 0.0333 + random.random() * 0.016667,
            "spatter_post_color": True,
        },
        "post": lambda settings: [Effect("spatter", color=settings["spatter_post_color"])]
    },

    "spatter-final": {
        "settings": lambda: {
            "speed": 0.0333 + random.random() * 0.016667,
            "spatter_final_color": True,
        },
        "final": lambda settings: [Effect("spatter", color=settings["spatter_final_color"])]
    },

    "splork": {
        "layers": ["voronoi", "posterize", "distressed"],
        "settings": lambda: {
            "color_space": color.rgb,
            "dist_metric": distance.chebyshev,
            "distrib": distrib.ones,
            "freq": 33,
            "mask": mask.bank_ocr,
            "palette_on": True,
            "posterize_levels": random.randint(1, 3),
            "spline_order": interp.cosine,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 1,
            "voronoi_point_freq": 2,
            "voronoi_refract": 0.125,
        },
        "ai": {
            "prompt": "high contrast design with distorted symbols and geometric shapes, alien glyphs and graffiti",
            "image_strength": 0.5,
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "spooky-ticker": {
        "final": lambda settings: [Effect("spooky_ticker")]
    },

    "stackin-bricks": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "dist_metric": distance.triangular,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_inverse": True,
            "voronoi_point_freq": 10,
        },
        "ai": {
            "prompt": "stacked cubes, qbert, high detail",
            "style_preset": "isometric"
        }
    },

    "starfield": {
        "layers": ["multires-low", "brightness-post", "nebula", "contrast-post", "lens", "grain", "vignette-dark", "contrast-final"],
        "settings": lambda: {
            "brightness_post": -0.075,
            "color_space": color.hsv,
            "contrast_post": 2.0,
            "distrib": distrib.exp,
            "freq": random.randint(400, 500),
            "hue_range": 1.0,
            "mask": mask.sparser,
            "mask_static": True,
            "palette_on": False,
            "saturation": 0.75,
            "spline_order": interp.linear,
        },
        "ai": {
            "prompt": "space telescope imagery, stars in the night sky, lens flare",
            "style_preset": "photographic",
        }
    },

    "stray-hair": {
        "final": lambda settings: [Effect("stray_hair")]
    },

    "string-theory": {
        "layers": ["multires-low", "erosion-worms", "bloom", "lens"],
        "settings": lambda: {
            "color_space": color.rgb,
            "erosion_worms_alpha": 0.875 + random.random() * 0.125,
            "erosion_worms_contraction": 4.0 + random.random() * 2.0,
            "erosion_worms_density": 0.25 + random.random() * 0.125,
            "erosion_worms_iterations": random.randint(1250, 2500),
            "octaves": random.randint(2, 4),
            "palette_on": False,
            "ridges": False,
        },
        "ai": {
            "prompt": "flow field, fractal flame, cosmic string",
        }
    },

    "subpixelator": {
        "layers": ["basic", "subpixels", "funhouse"],
        "settings": lambda: {
            "palette_on": False,
        },
        "ai": {
            "prompt": "rgb subpixels, red green blue pixel elements, monitor macro",
            "image_strength": 0.75,
            "cfg_scale": 30,
        }
    },

    "subpixels": {
        "post": lambda settings: [
            Effect("glyph_map",
                   mask=random_member(mask.rgb_members()),
                   zoom=random_member([8, 16]))
        ]
    },

    "symmetry": {
        "layers": ["basic"],
        "settings": lambda: {
            "corners": True,
            "freq": [2, 2],
        },
        "ai": {
            "prompt": "soft blended colors, four-way symmetry, natural symmetry, kaleidoscope"
        }
    },

    "swerve-h": {
        "settings": lambda: {
            "swerve_h_displacement": 0.5 + random.random() * 0.5,
            "swerve_h_freq": [random.randint(2, 5), 1],
            "swerve_h_octaves": 1,
            "swerve_h_spline_order": interp.bicubic
        },
        "post": lambda settings: [
            Effect("warp",
                   displacement=settings["swerve_h_displacement"],
                   freq=settings["swerve_h_freq"],
                   octaves=settings["swerve_h_octaves"],
                   spline_order=settings["swerve_h_spline_order"])
        ]
    },

    "swerve-v": {
        "settings": lambda: {
            "swerve_v_displacement": 0.5 + random.random() * 0.5,
            "swerve_v_freq": [1, random.randint(2, 5)],
            "swerve_v_octaves": 1,
            "swerve_v_spline_order": interp.bicubic
        },
        "post": lambda settings: [
            Effect("warp",
                   displacement=settings["swerve_v_displacement"],
                   freq=settings["swerve_v_freq"],
                   octaves=settings["swerve_v_octaves"],
                   spline_order=settings["swerve_v_spline_order"])
        ]
    },

    "teh-matrex-haz-u": {
        "layers": ["glyph-map", "bloom", "contrast-post", "lens", "crt"],
        "settings": lambda: {
            "contrast_post": 2.0,
            "freq": (random.randint(2, 3), random.randint(24, 48)),
            "glyph_map_colorize": True,
            "glyph_map_mask": random_member(mask.glyph_members()),
            "glyph_map_zoom": random.randint(2, 6),
            "hue_rotation": 0.4 + random.random() * 0.2,
            "hue_range": 0.25,
            "lattice_drift": 1,
            "mask": mask.dropout,
            "spline_order": interp.cosine,
        },
        "ai": {
            "prompt": "matrix computer code, hacker programming language, sci-fi font, language glyphs",
        }
    },

    "tensor-tone": {
        "post": lambda settings: [
            Effect("glyph_map",
                   mask=mask.halftone,
                   colorize=coin_flip())
        ]
    },

    "tensorflower": {
        "layers": ["symmetry", "voronoi", "vortex", "bloom", "lens"],
        "settings": lambda: {
            "color_space": color.rgb,
            "dist_metric": distance.euclidean,
            "palette_on": False,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_nth": 0,
            "voronoi_point_corners": True,
            "voronoi_point_distrib": point.square,
            "voronoi_point_freq": 2,
            "vortex_range": random.randint(8, 25),
        },
        "ai": {
            "prompt": "parabolic tiling, flower with four petals, four-way symmetry",
            "image_strength": 0.875,
            "cfg_scale": 25,
        }
    },

    "terra-terribili": {
        "layers": ["multires-ridged", "shadow", "lens", "grain"],
        "settings": lambda: {
            "hue_range": 0.5 + random.random() * 0.5,
            "lattice_drift": random.random(),
            "octaves": 10,
            "palette_on": True
        },
        "ai": {
            "prompt": "terra terribili, satellite photography, aerial photography, high detail, eroded sci-fi terrain, high-relief, geomorphology, molten hellscape, orbiter imagery, cratered rocky surface, fantasy rpg zone, scarred, io, volcanoes",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "test-pattern": {
        "layers": ["basic", "posterize", "swerve-h", "pixel-sort", "snow", "be-kind-rewind", "lens"],
        "settings": lambda: {
            "brightness_distrib": distrib.ones,
            "distrib": random_member([m for m in distrib if distrib.is_scan(m)]),
            "freq": 1,
            "hue_range": 0.5 + random.random() * 1.5,
            "pixel_sort_angled": False,
            "pixel_sort_darkest": False,
            "posterize_levels": random.randint(2, 4),
            "saturation_distrib": distrib.ones,
            "swerve_h_displacement": 0.25 + random.random() * 0.25,
            "vignette_alpha": 0.05 + random.random() * 0.025,
        },
        "ai": {
            "prompt": "broadcast tv test pattern, visual distortion pattern, wavy test card, television test signal, color bars",
            "image_strength": 0.5,
            "cfg_scale": 25,
        }
    },

    "texture": {
        "final": lambda settings: [Effect("texture")],
    },

    "the-arecibo-response": {
        "layers": ["value-mask", "snow", "crt"],
        "settings": lambda: {
            "freq": random.randint(21, 105),
            "mask": mask.arecibo,
            "mask_repeat": random.randint(2, 6),
        },
        "ai": {
            "prompt": "response to the arecibo signal, alien message, alien language, dna double helix, nucleotides"
        }
    },

    "the-data-must-flow": {
        "layers": ["basic", "worms", "derivative-post", "brightness-post", "contrast-post", "glowing-edges", "maybe-rotate", "bloom", "lens"],
        "settings": lambda: {
            "color_space": color.rgb,
            "contrast_post": 2.0,
            "freq": [3, 1],
            "worms_alpha": 0.95 + random.random() * 0.125,
            "worms_behavior": worms.obedient,
            "worms_density": 2.0 + random.random(),
            "worms_duration": 1,
            "worms_stride": 8,
            "worms_stride_deviation": 6,
        },
        "ai": {
            "prompt": "flow field, abstract representation of data flowing and converging",
        }
    },

    "the-inward-spiral": {
        "layers": ["voronoi", "worms", "brightness-post", "contrast-post", "bloom", "lens"],
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "freq": random.randint(12, 24),
            "voronoi_alpha": 1.0 - (random.randint(0, 1) * random.random() * 0.125),
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 0,
            "voronoi_point_freq": 1,
            "worms_alpha": 1,
            "worms_behavior": random_member([worms.obedient, worms.unruly, worms.crosshatch]),
            "worms_duration": random.randint(1, 4),
            "worms_density": 500,
            "worms_kink": random.randint(6, 24),
        },
        "ai": {
            "prompt": "flow field, spiralling path towards center, center distance",
        }
    },

    "time-crystal": {
        "layers": ["periodic-distance", "reflect-post", "grain", "saturation", "crt"],
        "settings": lambda: {
            "distrib": random_member([distrib.center_triangle, distrib.center_hexagon]),
            "hue_range": 2.0 + random.random(),
            "freq": 1,
            "reflect_range": 2.0 + random.random(),
        },
        "ai": {
            "prompt": "light refracted through a clear crystal with prismatic colors",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
        }
    },

    "time-doughnuts": {
        "layers": ["periodic-distance", "funhouse", "posterize", "grain", "saturation", "scanline-error", "crt"],
        "settings": lambda: {
            "distrib": distrib.center_circle,
            "freq": random.randint(2, 3),
            "posterize_levels": 2,
            "speed": 0.05,
            "warp_octaves": 2,
            "warp_range": 0.1 + random.random() * 0.05,
            "warp_signed_range": True,
        },
        "ai": {
            "prompt": "distorted waves of solid color emanate from the center of the image"
        }
    },

    "timeworms": {
        "layers": ["basic", "reflect-octaves", "worms", "density-map", "bloom", "lens"],
        "settings": lambda: {
            "freq": random.randint(4, 18),
            "mask": mask.sparse,
            "mask_static": True,
            "octaves": random.randint(1, 3),
            "reflect_range": random.randint(0, 1) * random.random() * 2,
            "saturation": 0,
            "spline_order": random_member([m for m in interp if m != interp.bicubic]),
            "worms_alpha": 1,
            "worms_behavior": worms.obedient,
            "worms_density": 0.25,
            "worms_duration": 10,
            "worms_stride": 2,
            "worms_kink": 0.25 + random.random() * 2.5,
        },
        "ai": {
            "prompt": "flow field, branching and converging timelines",
        }
    },

    "tint": {
        "settings": lambda: {
            "tint_alpha": 0.125 + random.random() * 0.05,
        },
        "final": lambda settings: [Effect("tint", alpha=settings["tint_alpha"])]
    },

    "trench-run": {
        "layers": ["periodic-distance", "posterize", "sobel", "invert", "scanline-error", "crt"],
        "settings": lambda: {
            "distrib": distrib.center_square,
            "hue_range": 0.1,
            "hue_rotation": random.random(),
            "posterize_levels": 1,
            "speed": 1.0,
        },
        "ai": {
            "prompt": "tie fighter pilot's hud view flying through the death star's ravine, vector display, trench"
        }
    },

    "tri-hard": {
        "layers": ["voronoi", "posterize-outline", "maybe-rotate", "grain", "saturation"],
        "settings": lambda: {
            "dist_metric": random_member([distance.octagram, distance.triangular, distance.hexagram]),
            "hue_range": 0.125 + random.random(),
            "posterize_levels": 6,
            "voronoi_alpha": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_point_freq": random.randint(8, 10),
            "voronoi_refract": 0.333 + random.random() * 0.333,
            "voronoi_refract_y_from_offset": False,
        },
        "ai": {
            "prompt": "stylized psychedelic pattern with shapes and outlined regions of solid color",
            "image_strength": 0.5,
        }
    },

    "tribbles": {
        "layers": ["voronoi", "funhouse", "invert", "contrast-post", "worms", "maybe-rotate", "lens"],
        "settings": lambda: {
            "color_space": random_member([color.hsv, color.oklab]),
            "dist_metric": distance.euclidean,
            "hue_range": 0.5 + random.random() * 2.5,
            "octaves": random.randint(1, 4),
            "palette_on": False,
            "saturation": 0.375 + random.random() * 0.5,
            "voronoi_alpha": 0.625 + random.random() * 0.125,
            "voronoi_diagram_type": voronoi.range_regions,
            "voronoi_point_distrib": point.h_hex,
            "voronoi_point_drift": random.random() * 0.5,
            "voronoi_point_freq": random.randint(2, 10),
            "voronoi_nth": 0,
            "warp_freq": random.randint(2, 4),
            "warp_octaves": random.randint(2, 4),
            "warp_range": 0.05 + random.random() * 0.0025,
            "worms_alpha": 0.625 + random.random() * 0.125,
            "worms_behavior": worms.unruly,
            "worms_density": random.randint(500, 2000),
            "worms_drunkenness": random.random() * 0.125,
            "worms_duration": 0.25 + random.random() * 0.25,
            "worms_kink": 0.875 + random.random() * 0.25,
            "worms_stride": 0.75 + random.random() * 0.25,
            "worms_stride_deviation": 0.375 + random.random() * 0.25,
        },
        "generator": lambda settings: {
            "freq": [settings["voronoi_point_freq"]] * 2,
        },
        "ai": {
            "prompt": "tribbles, furry, fuzzy, fluffy, puffs",
            "image_strength": 0.875,
            "cfg_scale": 30,
            "style_preset": "photographic",
        }
    },

    "trominos": {
        "layers": ["value-mask", "posterize", "sobel", "maybe-rotate", "invert", "bloom", "crt", "lens"],
        "settings": lambda: {
            "mask": mask.tromino,
            "mask_repeat": random.randint(6, 12),
            "posterize_levels": random.randint(1, 4),
            "spline_order": random_member([interp.constant, interp.cosine]),
        },
        "ai": {
            "prompt": "a grid of shapes inspired by tetrominos, geometric shapes inspired by \"Tetris\""
        }
    },

    "truchet-maze": {
        "layers": ["value-mask", "posterize", "maybe-rotate", "bloom", "crt"],
        "settings": lambda: {
            "angle": random_member([0, 45, random.randint(0, 360)]),
            "mask": random_member([mask.truchet_lines, mask.truchet_curves]),
            "mask_repeat": random.randint(20, 40),
            "posterize_levels": random.randint(1, 4),
        },
        "ai": {
            "prompt": "maze generated with truchet tiles, `10 PRINT CHR$(205.5+RND(1)); : GOTO 10`",
            "image_strength": .625,
            "cfg_scale": 25,
        }
    },

    "turbulence": {
        "layers": ["basic-water", "periodic-refract", "refract-post", "lens", "contrast-post"],
        "settings": lambda: {
            "freq": random.randint(2, 3),
            "hue_range": 2.0,
            "hue_rotation": random.random(),
            "octaves": 3,
            "refract_range": 0.025 + random.random() * 0.0125,
            "refract_y_from_offset": False,
            "value_distrib": distrib.center_circle,
            "value_freq": 1,
            "value_refract_range": 0.05 + random.random() * 0.025,
            "speed": -0.05,
        },
        "ai": {
            "prompt": "colorful distorted ripples emanating from the center of the image",
            "style_preset": "photographic",
        }
    },

    "twisted": {
        "layers": ["basic", "worms"],
        "settings": lambda: {
            "freq": random.randint(6, 12),
            "hue_range": 0.0,
            "ridges": True,
            "saturation": 0.0,
            "worms_density": random.randint(125, 250),
            "worms_duration": 1.0 + random.random() * 0.5,
            "worms_quantize": True,
            "worms_stride": 1.0,
            "worms_stride_deviation": 0.5,
        },
        "ai": {
            "prompt": "flow field with quantized direction",
        }
    },

    "unicorn-puddle": {
        "layers": ["multires", "reflect-octaves", "refract-post", "random-hue", "bloom", "lens"],
        "settings": lambda: {
            "color_space": color.oklab,
            "distrib": distrib.uniform,
            "freq": 2,
            "hue_range": 2.0 + random.random(),
            "lattice_drift": 1.0,
            "palette_on": False,
            "reflect_range": 0.5 + random.random() * 0.25,
            "refract_range": 0.5 + random.random() * 0.25,
        },
        "ai": {
            "prompt": "melted paint, melting colors, unmixed",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "unmasked": {
        "layers": ["basic", "sobel", "invert", "reindex-octaves", "maybe-rotate", "bloom", "lens"],
        "settings": lambda: {
            "distrib": distrib.uniform,
            "freq": random.randint(3, 5),
            "mask": random_member(mask.procedural_members()),
            "octave_blending": blend.alpha,
            "octaves": random.randint(2, 4),
            "reindex_range": 1 + random.random() * 1.5,
        },
        "ai": {
            "prompt": "a noodly noise field processed with sobel shading",
        }
    },

    "value-mask": {
        "layers": ["basic"],
        "settings": lambda: {
            "distrib": distrib.ones,
            "mask": random_member(mask),
            "mask_repeat": random.randint(2, 8),
            "spline_order": random_member([m for m in interp if m != interp.bicubic])
        },
        "generator": lambda settings: {
            "freq": [int(i * settings["mask_repeat"]) for i in masks.mask_shape(settings["mask"])[0:2]],
        },
        "ai": {
            "prompt": "generative noise with a masked-out pattern",
        }
    },

    "value-refract": {
        "settings": lambda: {
            "value_freq": random.randint(2, 4),
            "value_refract_range": 0.125 + random.random() * 0.0625,
        },
        "post": lambda settings: [
            Effect("value_refract",
                   displacement=settings["value_refract_range"],
                   distrib=settings.get("value_distrib", distrib.uniform),
                   freq=settings["value_freq"])
        ]
    },

    "vaseline": {
        "settings": lambda: {
            "vaseline_alpha": 0.375 + random.random() * 0.1875
        },
        "final": lambda settings: [Effect("vaseline", alpha=settings["vaseline_alpha"])]
    },

    "vectoroids": {
        "layers": ["voronoi", "derivative-post", "glowing-edges", "bloom", "crt", "lens"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "distrib": distrib.ones,
            "freq": 40,
            "mask": mask.sparse,
            "mask_static": True,
            "spline_order": interp.constant,
            "voronoi_diagram_type": voronoi.color_regions,
            "voronoi_nth": 0,
            "voronoi_point_freq": 15,
            "voronoi_point_drift": 0.25 + random.random() * 0.75,
        },
        "ai": {
            "prompt": "vector display reminiscent of classic arcade game \"asteroids\", vectorized shapes, glowing outlines on a black background",
            "image_strength": 0.875,
            "style_preset": "neon-punk",
        }
    },

    "veil": {
        "layers": ["voronoi", "fractal-seed"],
        "settings": lambda: {
            "dist_metric": random_member([distance.manhattan, distance.octagram, distance.triangular]),
            "voronoi_diagram_type": random_member([voronoi.color_range, voronoi.range]),
            "voronoi_inverse": True,
            "voronoi_point_distrib": random_member(point.grid_members()),
            "voronoi_point_freq": random.randint(2, 3),
            "worms_behavior": worms.random,
            "worms_kink": 0.5 + random.random(),
            "worms_stride": random.randint(48, 96),
        },
        "ai": {
            "prompt": "flow field, fractal flame"
        }
    },

    "vibe": {
        "layers": ["basic", "reflect-post", "posterize-outline", "grain"],
        "settings": lambda: {
            "brightness_distrib": None,
            "color_space": color.oklab,
            "lattice_drift": 1.0,
            "palette_on": False,
            "reflect_range": 0.5 + random.random() * 0.5,
        },
        "ai": {
            "prompt": "psychedelic fractal imagery, vibey, subdued mood",
            "image_strength": 0.5,
            "cfg_scale": 20,
        }
    },

    "vignette-bright": {
        "settings": lambda: {
            "vignette_alpha": 0.333 + random.random() * 0.333,
            "vignette_brightness": 1.0,
        },
        "final": lambda settings: [
            Effect("vignette",
                   alpha=settings["vignette_alpha"],
                   brightness=settings["vignette_brightness"])
        ]
    },

    "vignette-dark": {
        "settings": lambda: {
            "vignette_alpha": 0.5 + random.random() * 0.25,
            "vignette_brightness": 0.0,
        },
        "final": lambda settings: [
            Effect("vignette",
                   alpha=settings["vignette_alpha"],
                   brightness=settings["vignette_brightness"])
        ]
    },

    "voronoi": {
        "layers": ["basic"],
        "settings": lambda: {
            "dist_metric": random_member(distance.all()),
            "voronoi_alpha": 1.0,
            "voronoi_diagram_type": random_member([t for t in voronoi if t != voronoi.none]),
            "voronoi_sdf_sides": random.randint(2, 8),
            "voronoi_inverse": False,
            "voronoi_nth": random.randint(0, 2),
            "voronoi_point_corners": False,
            "voronoi_point_distrib": point.random if coin_flip() else random_member(point, mask.nonprocedural_members()),
            "voronoi_point_drift": 0.0,
            "voronoi_point_freq": random.randint(8, 15),
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
                   refract_y_from_offset=settings["voronoi_refract_y_from_offset"],
                   sdf_sides=settings["voronoi_sdf_sides"])
        ],
        "ai": {
            "prompt": "voronoi diagram overlayed on a field of value noise",
            "image_strength": .75,
        },
    },

    "voronoi-refract": {
        "layers": ["voronoi"],
        "settings": lambda: {
            "palette_on": False,
            "voronoi_refract": 0.25 + random.random() * 0.75
        },
        "ai": {
            "prompt": "value noise warped and deformed by a voronoi diagram, refracted colors",
            "image_strength": .75,
        },
    },

    "vortex": {
        "settings": lambda: {
            "vortex_range": random.randint(16, 48)
        },
        "post": lambda settings: [Effect("vortex", displacement=settings["vortex_range"])]
    },

    "warped-cells": {
        "layers": ["voronoi", "ridge", "funhouse", "bloom", "grain"],
        "settings": lambda: {
            "dist_metric": random_member(distance.absolute_members()),
            "voronoi_alpha": 0.666 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.color_range,
            "voronoi_nth": 0,
            "voronoi_point_distrib": random_member(point, mask.nonprocedural_members()),
            "voronoi_point_freq": random.randint(6, 10),
            "warp_range": 0.25 + random.random() * 0.25,
        },
        "ai": {
            "prompt": "warped and distorted voronoi cells, colorful stretched shapes",
            "image_strength": .875,
            "cfg_scale": 25,
        }
    },

    "whatami": {
        "layers": ["voronoi", "reindex-octaves", "reindex-post", "grain", "saturation"],
        "settings": lambda: {
            "freq": random.randint(7, 9),
            "hue_range": random.randint(3, 12),
            "reindex_range": 1.5 + random.random() * 1.5,
            "voronoi_alpha": 0.75 + random.random() * 0.125,
            "voronoi_diagram_type": voronoi.color_range,
        },
        "ai": {
            "prompt": "psychedelic fractal pattern with repeated bands of color",
            "image_strength": 0.375,
            "cfg_scale": 25,
        }
    },

    "wild-kingdom": {
        "layers": ["basic", "funhouse", "posterize-outline", "shadow", "maybe-invert", "lens", "grain", "nudge-hue"],
        "settings": lambda: {
            "color_space": color.rgb,
            "freq": 20,
            "lattice_drift": 0.333,
            "mask": mask.sparse,
            "mask_static": True,
            "palette_on": False,
            "posterize_levels": random.randint(2, 6),
            "ridges": True,
            "spline_order": interp.cosine,
            "vaseline_alpha": 0.1 + random.random() * 0.05,
            "vignette_alpha": 0.1 + random.random() * 0.05,
            "warp_octaves": 3,
            "warp_range": 0.0333,
        },
        "ai": {
            "prompt": "microscopic view of single-celled organisms, glowing amoeba blobs, are they having a party?",
            "image_strength": 0.5,
            "cfg_scale": 25,
            "style_preset": "photographic",
            "model": "stable-diffusion-xl-1024-v1-0",
        }
    },

    "woahdude": {
        "layers": ["wobble", "voronoi", "sine-octaves", "refract-post", "bloom", "saturation", "lens"],
        "settings": lambda: {
            "dist_metric": distance.euclidean,
            "freq": random.randint(3, 5),
            "hue_range": 2,
            "lattice_drift": 1,
            "refract_range": 0.0005 + random.random() * 0.00025,
            "saturation_final": 1.5,
            "sine_range": random.randint(40, 60),
            "speed": 0.025,
            "tint_alpha": 0.05 + random.random() * 0.025,
            "voronoi_refract": 0.333 + random.random() * 0.333,
            "voronoi_diagram_type": voronoi.range,
            "voronoi_nth": 0,
            "voronoi_point_distrib": random_member(point.circular_members()),
            "voronoi_point_freq": 6,
        },
        "ai": {
            "prompt": "psychedelic fractal imagery",
        }
    },

    "wobble": {
        "post": lambda settings: [Effect("wobble")]
    },

    "wormhole": {
        "settings": lambda: {
            "wormhole_kink": 1.0 + random.random() * 0.5,
            "wormhole_stride": 0.05 + random.random() * 0.025
        },
        "post": lambda settings: [
            Effect("wormhole",
                   kink=settings["wormhole_kink"],
                   input_stride=settings["wormhole_stride"])
        ]
    },

    "worms": {
        "settings": lambda: {
            "worms_alpha": 0.75 + random.random() * 0.25,
            "worms_behavior": random_member(worms.all()),
            "worms_density": random.randint(250, 500),
            "worms_drunkenness": 0.0,
            "worms_duration": 1.0 + random.random() * 0.5,
            "worms_kink": 1.0 + random.random() * 0.5,
            "worms_quantize": False,
            "worms_stride": 0.75 + random.random() * 0.5,
            "worms_stride_deviation": random.random() + 0.5
        },
        "post": lambda settings: [
            Effect("worms",
                   alpha=settings["worms_alpha"],
                   behavior=settings["worms_behavior"],
                   density=settings["worms_density"],
                   drunkenness=settings["worms_drunkenness"],
                   duration=settings["worms_duration"],
                   kink=settings["worms_kink"],
                   quantize=settings["worms_quantize"],
                   stride=settings["worms_stride"],
                   stride_deviation=settings["worms_stride_deviation"])
        ]
    },

    "wormstep": {
        "layers": ["basic", "worms"],
        "settings": lambda: {
            "corners": True,
            "lattice_drift": coin_flip(),
            "octaves": random.randint(1, 3),
            "palette_name": None,
            "worms_alpha": 0.5 + random.random() * 0.5,
            "worms_behavior": worms.chaotic,
            "worms_density": 500,
            "worms_kink": 1.0 + random.random() * 4.0,
            "worms_stride": 8.0 + random.random() * 4.0,
        },
        "ai": {
            "prompt": "flow field, fractal flame, noise contours",
        }
    },

    "writhe": {
        "layers": ["multires-alpha", "octave-warp-octaves", "brightness-post", "shadow", "grain", "lens"],
        "settings": lambda: {
            "color_space": color.oklab,
            "ridges": True,
            "speed": 0.025,
            "warp_freq": [random.randint(2, 3), random.randint(2, 3)],
            "warp_range": 5.0 + random.random() * 2.5,
        },
        "ai": {
            "prompt": "a writhing mass of overlapping noise, seething chaos",
        }
    },

    "zeldo": {
        "layers": ["glyph-map", "posterize", "crt"],
        "settings": lambda: {
            "freq": random.randint(3, 9),
            "glyph_map_colorize": True,
            "glyph_map_mask": mask.mcpaint,
            "glyph_map_zoom": random.randint(2, 4),
            "spline_order": random_member([interp.constant, interp.linear]),
        },
        "ai": {
            "prompt": "8-bit tiled sprites, retro rpg map, classic arcade game, fantasy dungeon",
            "image_strength": 0.625,
            "cfg_scale": 30,
            "style_preset": "pixel-art",
        }
    },

}

Preset = functools.partial(Preset, presets=PRESETS())
