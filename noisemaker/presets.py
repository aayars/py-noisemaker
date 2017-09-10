import random

import noisemaker.generators as generators

PRESETS = {
    "2d-chess": {
        "kwargs": {
            "distrib": "ones",
            "freq": 8,
            "mask": "chess",
            "point_distrib": "square",
            "point_freq": 8,
            "spline_order": 0,
            "voronoi_alpha": 0.5,
            "with_voronoi": 2,
        }
    },

    "acid-grid": {
        "kwargs": {
            "invert": 1,
            "lattice_drift": 1,
            "point_distrib": "circular",
            "point_freq": 4,
            "point_generations": 2,
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0.5,
            "voronoi_func": 3,
            "voronoi_nth": 1,
            "warp_range": 0.25,
            "warp_octaves": 1,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "with_voronoi": 2,
        }
    },

    "alien-terrain": {
        "kwargs": {
            "deriv": 1,
            "deriv_alpha": 0.5,
            "freq": 5,
            "hsv_rotation": 0,
            "hsv_saturation": 0.4,
            "lattice_drift": 1,
            "octaves": 3,
            "point_freq": 10,
            "point_distrib": "random",
            "ridges": True,
            "shadow": 1,
            "sin": 1.4,
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0.5,
            "with_bloom": 0.5,
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_bg": 75,
            "worms_kink": 48,
            "worms_duration": 0.4,
            "worms_density": 500,
        }
    },

    "aztec-waffles": {
        "kwargs": {
            "freq": 7,
            "point_freq": 4,
            "point_generations": 2,
            "point_distrib": "circular",
            "posterize_levels": 12,
            "reflect_range": 2.06,
            "spline_order": 0,
            "voronoi_func": 2,
            "voronoi_nth": 2,
            "with_outline": 3,
            "with_voronoi": 1,
        }
    },
     
    "badlands": {
        "kwargs": {
            "deriv": 2,
            "deriv_alpha": 0.5,
            "hsv_rotation": 0.95,
            "hsv_saturation": 0.25,
            "lattice_drift": 1,
            "octaves": 3,
            "point_distrib": "random",
            "point_freq": 10,
            "ridges": True,
            "shadow": 1,
            "sharpen": 1,
            "sin": -0.5,
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0.333,
            "with_bloom": 0.333,
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_bg": 125,
            "worms_density": 500,
            "worms_duration": 0.3333,
            "worms_kink": 64,
        }
    },
     
    "bringing-hexy-back": {
        "kwargs": {
            "lattice_drift": 1,
            "point_distrib": "v_hex",
            "point_freq": 10,
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
            "with_voronoi": 5,
        }
    },
     
    "circulent": {
        "kwargs": {
            "freq": 2,
            "point_distrib": "spiral",
            "point_freq": 6,
            "voronoi_nth": 2,
            "with_voronoi": 2,
            "with_wormhole": True,
            "wormhole_kink": 10,
            "wormhole_stride": 0.005,
        }
    },
     
    "conjoined": {
        "kwargs": {
            "lattice_drift": 1,
            "point_distrib": "circular",
            "point_freq": 5,
            "voronoi_alpha": 0.5,
            "voronoi_nth": 1,
            "voronoi_refract": 11,
            "with_bloom": 0.5,
            "with_voronoi": 2,
        }
    },
     
    "cubic": {
        "kwargs": {
            "freq": 2,
            "point_distrib": "concentric",
            "point_freq": 4,
            "voronoi_alpha": 0.5,
            "voronoi_nth": random.randint(4, 12),
            "with_bloom": 0.5,
            "with_outline": 1,
            "with_voronoi": 2,
        }
    },
     
    "death-star-plans": {
        "kwargs": {
            "channels": 1,
            "invert": 1,
            "octaves": 1,
            "point_freq": 10,
            "posterize_levels": 4,
            "voronoi_alpha": 1,
            "voronoi_func": 3,
            "voronoi_nth": 1,
            "with_sobel": 2,
            "with_voronoi": 1,
        }
    },
     
    "defocus": {
        "kwargs": {
            "freq": 12,
            "mask": [m.value for m in generators.ValueMask][random.randint(0, len(generators.ValueMask) - 1)],
            "octaves": 5,
            "sin": 10,
            "with_bloom": 0.5,
        }
    },
     
    "furry-swirls": {
        "kwargs": {
            "freq": 32,
            "hsv_range": 2,
            "point_freq": 10,
            "spline_order": 1,
            "voronoi_alpha": 0.75,
            "with_voronoi": 6,
            "with_worms": 1,
            "worms_density": 64,
            "worms_duration": 1,
            "worms_kink": 25,
        }
    },
     
    "furry-thorns": {
        "kwargs": {
            "point_distrib": "waffle",
            "point_generations": 2,
            "voronoi_inverse": True,
            "voronoi_nth": 9,
            "with_voronoi": 2,
            "with_worms": 3,
            "worms_density": 500,
            "worms_duration": 1.22,
            "worms_kink": 2.89,
            "worms_stride": 0.64,
            "worms_stride_deviation": 0.11,
        }
    },
     
    "graph-paper": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 12) * 2,
            "hsv_range": 0,
            "hsv_saturation": 0.27,
            "mask": "chess",
            "posterize_levels": 12,
            "spline_order": 0,
            "with_sobel": 2,
        }
    },
     
    "hex-machine": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": 4,
            "mask": "h_tri",
            "octaves": random.randint(4, 8),
            "post_deriv": 3,
            "sin": 12.5,
        }
    },
     
    "jovian-clouds": {
        "kwargs": {
            "point_freq": 10,
            "voronoi_refract": 2,
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_density": 500,
            "worms_duration": 0.5,
            "worms_kink": 96,
        }
    },
     
    "magic-squares": {
        "kwargs": {
            "channels": 3,
            "distrib": "uniform",
            "edges": 0.81,
            "freq": 12,
            "octaves": 3,
            "spline_order": 0,
        }
    },
     
    "misaligned": {
        "kwargs": {
            "freq": 16,
            "mask": "v_tri",
            "octaves": 8,
            "spline_order": 0,
            "with_outline": 1,
        }
    },
     
    "neon-cambrian": {
        "kwargs": {
            "hsv_range": 1,
            "invert": 1,
            "posterize_levels": 24,
            "with_aberration": 0.01,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "with_voronoi": 6,
            "with_wormhole": True,
            "wormhole_kink": 1,
            "wormhole_stride": 0.25,
        }
    },
     
    "neon-plasma": {
        "kwargs": {
            "channels": 3,
            "freq": 4,
            "octaves": 8,
            "ridges": True,
            "wavelet": True,
        }
    },
     
    "now": {
        "kwargs": {
            "channels": 3,
            "lattice_drift": 1,
            "octaves": 3,
            "spline_order": 0,
            "voronoi_refract": 2,
            "with_outline": 1,
            "with_voronoi": 6,
        }
    },
     
    "plaid": {
        "kwargs": {
            "deriv": 3,
            "distrib": "ones",
            "freq": 8,
            "mask": "chess",
            "octaves": 3,
        }
    },
     
    "political-map": {
        "kwargs": {
            "freq": 5,
            "hsv_saturation": 0.35,
            "lattice_drift": 1,
            "octaves": 2,
            "posterize_levels": 4,
            "warp_octaves": 8,
            "warp_range": 1,
            "with_bloom": 1,
            "with_outline": 1,
        },

        "post_kwargs": {
            "with_dither": 0.25,
        }
    },
     
    "quilty": {
        "kwargs": {
            "freq": 5,
            "point_distrib": "square",
            "point_freq": random.randint(3, 8),
            "spline_order": 0,
            "voronoi_func": 2,
            "voronoi_refract": 2,
            "with_voronoi": 1,
        }
    },
     
    "reef": {
        "kwargs": {
            "lattice_drift": 1,
            "point_distrib": "circular",
            "freq": random.randint(2, 8),
            "point_freq": random.randint(2, 8),
            "voronoi_alpha": 0.5,
            "voronoi_refract": random.randint(12, 24),
            "with_bloom": 0.5,
            "with_voronoi": 6,
        }
    },
     
    "refractal": {
        "kwargs": {
            "channels": 1,
            "invert": 1,
            "lattice_drift": 1,
            "point_freq": 10,
            "post_reflect_range": 128,
            "with_voronoi": 6,
        }
    },
     
    "rings": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 10),
            "hsv_range": random.random(),
            "hsv_saturation": 0.5,
            "mask": "waffle",
            "octaves": random.randint(2, 5),
            "sin": random.random() * 5.0,
            "with_outline": 1,
        }
    },
     
    "skeletal-lace": {
        "kwargs": {
            "lattice_drift": 1,
            "point_freq": 3,
            "voronoi_nth": 1,
            "voronoi_refract": 25,
            "with_voronoi": 6,
            "with_wormhole": True,
            "wormhole_stride": 0.01,
        }
    },
     
    "spiral-in-spiral": {
        "kwargs": {
            "point_distrib": "spiral",
            "point_freq": 10,
            "with_voronoi": 1,
            "with_worms": 1,
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": 10,
        }
    },
     
    "spiraltown": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 1,
            "reflect_range": 5,
            "spline_order": 2,
            "with_wormhole": True,
            "wormhole_kink": 10,
            "wormhole_stride": 0.0025,
        }
    },
     
    "square-stripes": {
        "kwargs": {
            "point_distrib": "v_hex",
            "point_freq": 2,
            "point_generations": 2,
            "voronoi_alpha": 0.78,
            "voronoi_func": 3,
            "voronoi_inverse": True,
            "voronoi_nth": 3,
            "voronoi_refract": 1.46,
            "with_voronoi": 2,
        }
    },
     
    "star-cloud": {
        "kwargs": {
            "deriv": 1,
            "freq": 2,
            "invert": 1,
            "point_freq": 10,
            "reflect_range": 2,
            "spline_order": 2,
            "voronoi_refract": 2.5,
            "with_sobel": 1,
            "with_voronoi": 6,
        }
    },
     
    "traceroute": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": 4,
            "mask": "v_tri",
            "octaves": 8,
            "with_worms": 1,
            "worms_density": 500,
            "worms_kink": 15,
        }
    },
     
    "triangular": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(1, 4) * 2,
            "mask": "h_tri",
            "octaves": random.randint(4, 10),
        }
    },
     
    "tribbles": {
        "kwargs": {
            "hsv_rotation": 0.4,
            "hsv_saturation": 0.333,
            "invert": 1,
            "octaves": 3,
            "point_distrib": "h_hex",
            "point_freq": 6,
            "ridges": True,
            "voronoi_alpha": 0.5,
            "warp_octaves": 7,
            "warp_range": 0.25,
            "with_bloom": 0.5,
            "with_voronoi": 5,
            "with_worms": 3,
            "worms_bg": 5,
            "worms_density": 500,
            "worms_duration": .5,
            "worms_stride_deviation": .25,
        }
    },
     
    "velcro": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 4,
            "reflect_range": 8,
            "spline_order": 2,
            "with_wormhole": True,
            "wormhole_stride": 0.025,
        }
    },
     
    "warped-grid": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 12) * 2,
            "hsv_range": 0,
            "hsv_saturation": 0.27,
            "invert": 1,
            "mask": "chess",
            "posterize_levels": 12,
            "spline_order": 0,
            "warp_interp": 3,
            "warp_freq": random.randint(2, 4),
            "warp_range": .25 + random.random() * .75,
            "warp_octaves": 1,
            "with_sobel": 2,
        }
    },

    "wireframe": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 0.11,
            "hsv_rotation": 1,
            "hsv_saturation": 0.48,
            "invert": 1,
            "lattice_drift": 0.68,
            "octaves": 2,
            "point_distrib": "v_hex",
            "point_freq": 10,
            "voronoi_alpha": 0.5,
            "voronoi_nth": 1,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "with_voronoi": 5,
        }
    },
     
    "wireframe-warped": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 0.11,
            "hsv_rotation": 1,
            "hsv_saturation": 0.48,
            "invert": 1,
            "lattice_drift": 0.68,
            "octaves": 2,
            "point_distrib": "v_hex",
            "point_freq": 10,
            "voronoi_alpha": 0.5,
            "voronoi_nth": 1,
            "warpInterp": 3,
            "warp_octaves": 3,
            "warp_range": 1.4,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "with_voronoi": 5,
        }
    },
     
    "web-of-lies": {
        "kwargs": {
            "point_distrib": "spiral",
            "point_freq": 10,
            "voronoi_alpha": 0.5,
            "voronoi_refract": 2,
            "with_bloom": 0.5,
            "with_voronoi": 1,
         }
     },
  
    "woahdude": {
        "kwargs": {
            "freq": 4,
            "hsv_range": 2,
            "lattice_drift": 1,
            "point_freq": 8,
            "sin": 100,
            "voronoi_alpha": 0.875,
            "voronoi_refract": 1,
            "with_voronoi": 1,
        }
    },
     
    "wooly-bully": {
        "kwargs": {
            "hsv_range": 1,
            "point_corners": True,
            "point_distrib": "circular",
            "point_freq": random.randint(1, 3),
            "point_generations": 2,
            "voronoi_func": 3,
            "voronoi_nth": 2,
            "voronoi_alpha": .5 + random.random() * .5,
            "with_voronoi": 2,
            "with_worms": 4,
            "worms_bg": 0.78,
            "worms_density": 346.75,
            "worms_duration": 2.2,
            "worms_kink": 6.47,
            "worms_stride": 2.5,
            "worms_stride_deviation": 1.25,
        }
    }
}

def load(preset_name):
    """
    """

    if preset_name == "random":
        preset = PRESETS.get(list(PRESETS)[random.randint(0, len(PRESETS) - 1)])

    else:
        preset = PRESETS.get(preset_name, {})

    return preset.get("kwargs", {}), preset.get("post_kwargs", {})