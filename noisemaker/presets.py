import random

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
            "worms_density": 2048,
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
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_kink": 64,
            "worms_duration": 0.3333,
            "worms_density": 2048,
            "point_freq": 10,
            "voronoi_refract": 0.333,
            "point_distrib": "random",
            "with_bloom": 0.333,
            "worms_bg": 125,
            "octaves": 3,
            "shadow": 1,
            "voronoi_alpha": 0.5,
            "ridges": True,
            "deriv": 2,
            "hsv_rotation": 0.95,
            "hsv_saturation": 0.25,
            "lattice_drift": 1,
            "deriv_alpha": 0.5,
            "sin": -0.5,
            "sharpen": 1,
        }
    },
     
    "bringing-hexy-back": {
        "kwargs": {
            "with_voronoi": 5,
            "point_distrib": "v_hex",
            "point_freq": 10,
            "lattice_drift": 1,
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
        }
    },
     
    "circulent": {
        "kwargs": {
            "with_voronoi": 2,
            "point_distrib": "spiral",
            "freq": 2,
            "with_wormhole": True,
            "point_freq": 6,
            "wormhole_stride": 0.005,
            "voronoi_nth": 2,
            "wormhole_kink": 10,
        }
    },
     
    "conjoined": {
        "kwargs": {
            "with_voronoi": 2,
            "point_distrib": "circular",
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
            "voronoi_refract": 11,
            "point_freq": 5,
            "lattice_drift": 1,
            "voronoi_nth": 1,
        }
    },
     
    "cubic": {
        "kwargs": {
            "with_voronoi": 2,
            "point_freq": 4,
            "voronoi_nth": random.randint(4, 12),
            "point_distrib": "concentric",
            "with_outline": 1,
            "freq": 2,
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
        }
    },
     
    "death-star-plans": {
        "kwargs": {
            "channels": 1,
            "octaves": 1,
            "with_voronoi": 1,
            "point_freq": 10,
            "voronoi_func": 3,
            "voronoi_nth": 1,
            "voronoi_alpha": 1,
            "with_sobel": 2,
            "posterize_levels": 4,
            "invert": 1,
        }
    },
     
    "defocus": {
        "kwargs": {
            "mask": "h_tri",
            "freq": 12,
            "octaves": 5,
            "sin": 10,
            "with_bloom": 0.5,
        }
    },
     
    "furry-swirls": {
        "kwargs": {
            "with_voronoi": 6,
            "point_freq": 10,
            "spline_order": 1,
            "with_worms": 1,
            "worms_kink": 25,
            "worms_density": 64,
            "worms_duration": 1,
            "voronoi_alpha": 0.75,
            "freq": 32,
            "hsv_range": 2,
        }
    },
     
    "furry-thorns": {
        "kwargs": {
            "with_worms": 3,
            "worms_density": 764.53,
            "worms_duration": 1.22,
            "worms_stride": 0.64,
            "worms_stride_deviation": 0.11,
            "worms_kink": 2.89,
            "with_voronoi": 2,
            "voronoi_inverse": True,
            "voronoi_nth": 9,
            "point_distrib": "waffle",
            "point_generations": 2,
        }
    },
     
    "graph-paper": {
        "kwargs": {
            "freq": 20,
            "spline_order": 0,
            "distrib": "ones",
            "mask": "chess",
            "corners": True,
            "hsv_range": 0.79,
            "hsv_rotation": 0.9,
            "hsv_saturation": 0.27,
            "posterize_levels": 12,
            "with_sobel": 2,
        }
    },
     
    "hex-machine": {
        "kwargs": {
            "freq": 4,
            "octaves": random.randint(4, 8),
            "mask": "h_tri",
            "distrib": "ones",
            "post_deriv": 3,
            "corners": True,
            "sin": 12.5,
        }
    },
     
    "hexception": {
        "kwargs": {
            "with_voronoi": 2,
            "point_freq": 6,
            "point_distrib": "v_hex",
            "voronoi_alpha": 0.875,
            "mask": "v_hex",
            "freq": 36,
            "distrib": "ones",
            "octaves": 3,
            "edges": 0.25,
            "sharpen": 0.5,
            "point_corners": True,
        }
    },
     
    "jovian-clouds": {
        "kwargs": {
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_kink": 96,
            "worms_duration": 0.5,
            "worms_density": 2048,
            "point_freq": 10,
            "voronoi_refract": 2,
        }
    },
     
    "magic-squares": {
        "kwargs": {
            "octaves": 3,
            "freq": 12,
            "spline_order": 0,
            "distrib": "uniform",
            "channels": 3,
            "edges": 0.81,
        }
    },
     
    "misaligned": {
        "kwargs": {
            "mask": "v_tri",
            "freq": 16,
            "octaves": 8,
            "spline_order": 0,
            "with_outline": 1,
        }
    },
     
    "neon-cambrian": {
        "kwargs": {
            "posterize_levels": 24,
            "with_wormhole": True,
            "wormhole_stride": 0.25,
            "wormhole_kink": 1,
            "with_sobel": 1,
            "invert": 1,
            "with_voronoi": 6,
            "hsv_range": 1,
            "with_aberration": 0.01,
            "with_bloom": 0.5,
        }
    },
     
    "neon-plasma": {
        "kwargs": {
            "freq": 4,
            "channels": 3,
            "ridges": True,
            "wavelet": True,
            "octaves": 8,
        }
    },
     
    "now": {
        "kwargs": {
            "channels": 3,
            "spline_order": 0,
            "octaves": 3,
            "lattice_drift": 1,
            "with_voronoi": 6,
            "voronoi_refract": 2,
            "with_outline": 1,
        }
    },
     
    "plaid": {
        "kwargs": {
            "mask": "chess",
            "distrib": "ones",
            "octaves": 3,
            "freq": 8,
            "deriv": 3,
        }
    },
     
    "political-map": {
        "kwargs": {
            "lattice_drift": 1,
            "with_bloom": 1,
            "warp_range": 1,
            "warp_octaves": 8,
            "octaves": 2,
            "hsv_saturation": 0.35,
            "posterize_levels": 4,
            "with_outline": 1,
            "freq": 5,
        },

        "post_kwargs": {
            "with_dither": 0.25,
        }
    },
     
    "quilty": {
        "kwargs": {
            "with_voronoi": 1,
            "point_distrib": "square",
            "point_freq": random.randint(3, 8),
            "voronoi_func": 2,
            "spline_order": 0,
            "freq": 5,
            "voronoi_refract": 2,
        }
    },
     
    "reef": {
        "kwargs": {
            "with_voronoi": 6,
            "point_distrib": "circular",
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
            "voronoi_refract": 17,
            "point_freq": 5,
            "lattice_drift": 1,
        }
    },
     
    "refractal": {
        "kwargs": {
            "with_voronoi": 6,
            "lattice_drift": 1,
            "post_reflect_range": 128,
            "channels": 1,
            "point_freq": 10,
            "invert": 1,
        }
    },
     
    "rings": {
        "kwargs": {
            "octaves": random.randint(2, 5),
            "freq": random.randint(4, 10),
            "hsv_range": random.random(),
            "hsv_saturation": 0.5,
            "distrib": "ones",
            "mask": "waffle",
            "sin": random.random() * 5.0,
            "with_outline": 1,
            "corners": True,
        }
    },
     
    "skeletal-lace": {
        "kwargs": {
            "with_voronoi": 6,
            "voronoi_refract": 25,
            "lattice_drift": 1,
            "voronoi_nth": 1,
            "with_wormhole": True,
            "wormhole_stride": 0.01,
            "point_freq": 3,
        }
    },
     
    "spiral-in-spiral": {
        "kwargs": {
            "with_voronoi": 1,
            "point_distrib": "spiral",
            "point_freq": 10,
            "with_worms": 1,
            "worms_kink": 10,
            "worms_density": 500,
            "worms_duration": 1,
        }
    },
     
    "spiraltown": {
        "kwargs": {
            "freq": 2,
            "spline_order": 2,
            "reflect_range": 5,
            "with_wormhole": True,
            "wormhole_stride": 0.0025,
            "hsv_range": 1,
            "wormhole_kink": 10,
        }
    },
     
    "square-stripes": {
        "kwargs": {
            "with_voronoi": 2,
            "voronoi_inverse": True,
            "voronoi_func": 3,
            "voronoi_nth": 3,
            "voronoi_alpha": 0.78,
            "voronoi_refract": 1.46,
            "point_freq": 2,
            "point_distrib": "v_hex",
            "point_generations": 2,
        }
    },
     
    "star-cloud": {
        "kwargs": {
            "freq": 2,
            "spline_order": 2,
            "reflect_range": 2,
            "deriv": 1,
            "with_voronoi": 6,
            "voronoi_refract": 2.5,
            "with_sobel": 1,
            "invert": 1,
            "point_freq": 10,
        }
    },
     
    "traceroute": {
        "kwargs": {
            "mask": "v_tri",
            "with_worms": 1,
            "worms_density": 1000,
            "freq": 4,
            "octaves": 8,
            "distrib": "ones",
            "corners": True,
            "worms_kink": 15,
        }
    },
     
    "triangular": {
        "kwargs": {
            "mask": "h_tri",
            "freq": random.randint(1, 4) * 2,
            "distrib": "ones",
            "octaves": random.randint(4, 10),
            "corners": True,
        }
    },
     
    "tribbles": {
        "kwargs": {
            "point_freq": 6,
            "with_voronoi": 5,
            "point_distrib": "h_hex",
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
            "invert": 1,
            "octaves": 3,
            "ridges": True,
            "warp_range": 0.25,
            "warp_octaves": 7,
            "with_worms": 3,
            "worms_density": 2000,
            "worms_duration": 0.5,
            "worms_bg": 5,
            "hsv_saturation": 0.333,
            "hsv_rotation": 0.4,
        }
    },
     
    "velcro": {
        "kwargs": {
            "freq": 2,
            "spline_order": 2,
            "reflect_range": 8,
            "with_wormhole": True,
            "wormhole_stride": 0.025,
            "hsv_range": 4,
        }
    },
     
    "victorian-fractal": {
        "kwargs": {
            "with_voronoi": 5,
            "point_freq": 6,
            "point_distrib": "h_hex",
            "voronoi_alpha": 0.5,
            "mask": "v_hex",
            "freq": 36,
            "octaves": 3,
            "ridges": True,
            "voronoi_func": 2,
            "sharpen": 1,
            "distrib": "ones",
            "invert": 1,
            "with_bloom": 0.5,
            "voronoi_nth": 2,
            "corners": True,
        },

        "post_kwargs": {
            "with_dither": 0.25,
        }
    },
     
    "wireframe": {
        "kwargs": {
            "octaves": 2,
            "freq": 2,
            "hsv_range": 0.11,
            "hsv_rotation": 1,
            "hsv_saturation": 0.48,
            "with_voronoi": 5,
            "voronoi_nth": 1,
            "voronoi_alpha": 0.5,
            "point_freq": 10,
            "point_distrib": "v_hex",
            "lattice_drift": 0.68,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "invert": 1,
        }
    },
     
    "wireframe-warped": {
        "kwargs": {
            "octaves": 2,
            "freq": 2,
            "hsv_range": 0.11,
            "hsv_rotation": 1,
            "hsv_saturation": 0.48,
            "with_voronoi": 5,
            "voronoi_nth": 1,
            "voronoi_alpha": 0.5,
            "point_freq": 10,
            "point_distrib": "v_hex",
            "lattice_drift": 0.68,
            "with_bloom": 0.5,
            "with_sobel": 1,
            "invert": 1,
            "warp_range": 1.4,
            "warp_octaves": 3,
            "warpInterp": 3,
        }
    },
     
    "web-of-lies": {
        "kwargs": {
            "with_voronoi": 1,
            "point_distrib": "spiral",
            "point_freq": 10,
            "voronoi_refract": 2,
            "voronoi_alpha": 0.5,
            "with_bloom": 0.5,
         }
     },
  
    "woahdude": {
        "kwargs": {
            "with_voronoi": 1,
            "voronoi_alpha": 0.875,
            "point_freq": 8,
            "sin": 100,
            "hsv_range": 2,
            "lattice_drift": 1,
            "voronoi_refract": 1,
            "freq": 4,
        }
    },
     
    "wooly-bully": {
        "kwargs": {
            "hsv_range": 1,
            "with_worms": 4,
            "worms_density": 346.75,
            "worms_duration": 2.2,
            "worms_stride": 0.6,
            "worms_stride_deviation": 2.6,
            "worms_kink": 6.47,
            "worms_bg": 0.78,
            "with_voronoi": 2,
            "voronoi_func": 3,
            "voronoi_nth": 2,
            "point_freq": 2,
            "point_distrib": "circular",
            "point_generations": 2,
            "point_corners": True,
        }
    }
}

def load(preset_name):
    """
    """

    preset = PRESETS.get(preset_name, {})

    return preset.get("kwargs", {}), preset.get("post_kwargs", {})