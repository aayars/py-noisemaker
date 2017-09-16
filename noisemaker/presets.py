""" Presets library for noisemaker/artmaker """

import random

import noisemaker.generators as generators
import noisemaker.points as points

PRESETS = {
    "2d-chess": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": 8,
            "mask": "chess",
            "point_corners": True,
            "point_distrib": "square",
            "point_freq": 8,
            "spline_order": 0,
            "voronoi_alpha": 0.5 + random.random() * .5,
            "voronoi_nth": 0 if random.random() < .5 else random.randint(0,63),
            "with_voronoi": 2 if random.random() < .5 else random.randint(1,5),
        }
    },

    "acid-grid": {
        "kwargs": {
            "invert": 1,
            "lattice_drift": 1,
            "point_distrib": [m.value for m in points.PointDistribution][random.randint(0, len(points.PointDistribution) - 1)],
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

    "alien-terrain-2": {
        "kwargs": {
            "deriv": 1,
            "deriv_alpha": .5,
            "hsv_saturation": .5,
            "invert": 1,
            "lattice_drift": 1,
            "octaves": 8,
            "shadow": .75,
            "with_bloom": .25,
        }
    },

    "aztec-waffles": {
        "kwargs": {
            "freq": 7,
            "invert": (random.random() < .5),
            "point_freq": random.randint(2,4),
            "point_generations": 2,
            "point_distrib": "circular",
            "posterize_levels": random.randint(6,18),
            "reflect_range": random.random() * 2,
            "spline_order": 0,
            "voronoi_func": random.randint(2,3),
            "voronoi_nth": random.randint(2,4),
            "with_outline": 3,
            "with_voronoi": random.randint(1, 5),
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
            "point_distrib": "v_hex" if random.random() < .5 else "v_hex",
            "point_freq": 10,
            "post_deriv": 0 if random.random() < .5 else random.randint(1,3),
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0 if random.random() < .5 else random.random(),
            "warp_octaves": 1,
            "warp_range": random.random() * .5,
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
            "with_voronoi": random.randint(1,5),
            "with_wormhole": True,
            "wormhole_kink": random.randint(8,12),
            "wormhole_stride": random.random() * .01,
        }
    },
     
    "conjoined": {
        "kwargs": {
            "hsv_range": random.random(),
            "lattice_drift": 1,
            "point_distrib": ([m.value for m in points.PointDistribution.circular_members()])[random.randint(0, len(points.PointDistribution.circular_members()) - 1)],
            "point_freq": random.randint(3,7),
            "post_deriv": 1,
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(0,4),
            "voronoi_refract": random.randint(6,12),
            "with_bloom": 0.25 + random.random() * .5,
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

    "deadlock": {
        "kwargs": {
           "hsv_range": random.random(),
           "hsv_rotation": random.random(),
           "hsv_saturation": random.random(),
           "point_corners": (random.random() < .5),
           "point_distrib": [m.value for m in points.PointDistribution.grid_members()][random.randint(0, len(points.PointDistribution.grid_members()) - 1)],
           "point_drift": 0 if random.random() < .5 else random.random(),
           "point_freq": 4,
           "point_generations": 2,
           "voronoi_func": random.randint(2,3),
           "voronoi_nth": random.randint(0, 15),
           "voronoi_alpha": .5 + random.random() * .5,
           "sin": random.random() * 2,
           "with_outline": 3,
           "with_voronoi": 1,
        }
    },
     
    "death-star-plans": {
        "kwargs": {
            "channels": 1,
            "invert": 1,
            "octaves": 1,
            "point_freq": random.randint(2, 4),
            "post_refract_range": random.randint(0,1),
            "posterize_levels": random.randint(3,5),
            "voronoi_alpha": 1,
            "voronoi_func": random.randint(2,3),
            "voronoi_nth": random.randint(1,3),
            "with_aberration": random.random() * .01,
            "with_sobel": 2,
            "with_voronoi": random.randint(1,4),
        },

        "post_kwargs": {
            "with_crt": True,
            "with_scan_error": True,
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

    "dla": {
        "kwargs": {
            "dla_padding": random.randint(2, 8),
            "hsv_range": 1.5,
            "point_distrib": [m.value for m in points.PointDistribution][random.randint(0, len(points.PointDistribution) - 1)],
            "point_freq": random.randint(2, 8),
            "voronoi_alpha": random.random(),
            "with_bloom": random.random(),
            "with_dla": .5 + random.random() * .5,
            "with_voronoi": 0 if random.random() < .5 else random.randint(1,5),
        }
    },
     
    "ears": {
        "kwargs": {
            "freq":22,
            "distrib": "uniform",
            "hsv_range": random.random() * 2.5,
            "mask": [m.value for m in generators.ValueMask if m.name != "chess"][random.randint(0, len(generators.ValueMask) - 2)],
            "with_worms": 3,
            "worms_bg": 0.88,
            "worms_density": 188.07,
            "worms_duration": 3.20,
            "worms_stride": 0.40,
            "worms_stride_deviation": 0.31,
            "worms_kink": 6.36,
        }
    },

    "eyes": {
        "kwargs": {
            "corners": True,
            "distrib": ["ones", "uniform"][random.randint(0,1)],
            "freq": 12 * random.randint(1,2),
            "hsv_range": random.random(),
            "invert": 1,
            "mask": [m.value for m in generators.ValueMask if m.name != "chess"][random.randint(0, len(generators.ValueMask) - 2)],
            "ridges": True,
            "shadow": 1,
            "spline_order": random.randint(2,3),
            "with_outline": 1,
            "warp_freq": 2,
            "warp_octaves": 1,
            "warp_range": random.randint(1,4),
        }
    },

    "fuzzy-squares": {
        "kwargs": {
            "corners": True,
            "freq": 20,
            "distrib": "uniform",
            "mask": [m.value for m in generators.ValueMask if m.name != "chess"][random.randint(0, len(generators.ValueMask) - 2)],
            "spline_order": 1,
            "with_worms": 4,
            "worms_bg": random.random(),
            "worms_density": 400,
            "worms_duration": 2.0,
            "worms_stride": .5 + random.random(),
            "worms_stride_deviation": 1.0 + random.random(),
            "worms_kink": 1 + random.random() * 2.5,
        }
    },

    "fuzzy-swirls": {
        "kwargs": {
            "freq": random.randint(2,32),
            "hsv_range": random.random() * 2,
            "point_freq": random.randint(8,10),
            "spline_order": random.randint(1,3),
            "voronoi_alpha": 0.5 * random.random() * .5,
            "with_voronoi": 6,
            "with_worms": random.randint(1,4),
            "worms_density": 64,
            "worms_duration": 1,
            "worms_kink": 25,
        }
    },
     
    "fuzzy-thorns": {
        "kwargs": {
            "point_freq": random.randint(2,4),
            "point_distrib": "waffle",
            "point_generations": 2,
            "voronoi_inverse": True,
            "voronoi_nth": random.randint(6,12),
            "with_voronoi": 2,
            "with_worms": random.randint(1,3),
            "worms_density": 500,
            "worms_duration": 1.22,
            "worms_kink": 2.89,
            "worms_stride": 0.64,
            "worms_stride_deviation": 0.11,
        }
    },
     
    "glom": {
        "kwargs": {
            "lattice_drift": 1,
            "reflect_range": random.randint(2,4),
            "refract_range": random.randint(2,4),
            "ridges": True,
            "warp_range": random.random() * .5,
            "warp_octaves": 1,
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "graph-paper": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 12) * 2,
            "hsv_range": 0,
            "hsv_rotation": random.random(),
            "hsv_saturation": 0.27,
            "invert": random.randint(0, 1),
            "mask": "chess",
            # "posterize_levels": 12,
            "spline_order": 0,
            "voronoi_alpha": .25 + random.random() * .75,
            "voronoi_refract": random.random() * 4,
            "with_aberration": random.random() * .02,
            "with_bloom": .5,
            "with_sobel": 2,
            "with_voronoi": 6,
        },

        "post_kwargs": {
            "with_crt": True,
            "with_scan_error": True,
        }
    },

    "hairy-diamond": {
        "kwargs": {
            "freq": random.randint(2,6),
            "hsv_range": random.random(),
            "hsv_rotation": random.random(),
            "hsv_saturation": random.random(),
            "point_corners": True,
            "point_distrib": "concentric",
            "point_freq": random.randint(3,6),
            "point_generations": 2,
            "voronoi_func": 2,
            "voronoi_inverse": True,
            "voronoi_alpha": .25 + random.random() * .5,
            "with_erosion_worms": True,
            "with_voronoi": 6,
        }
    },

    "halt-catch-fire": {
        "kwargs": {
            "freq": 2,
            "hsv_range": .05,
            "lattice_drift": 1,
            "octaves": random.randint(3, 5),
            "spline_order": 0,
            "with_aberration": .01 + random.random() * .01,
        },

        "post_kwargs": {
            "with_crt": True,
            "with_glitch": True,
            "with_scan_error": True,
            "with_snow": random.random() * .333,
        }
    },
     
    "hex-machine": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(1,3) * 2,
            "mask": "h_tri",
            "octaves": random.randint(5, 8),
            "post_deriv": 3,
            "sin": random.randint(-25, 25),
        }
    },

    "hsv-gradient": {
        "kwargs": {
            "freq": random.randint(2, 3),
            "hsv_range": .125 + random.random() * 2.0,
            "lattice_drift": random.random(),
        }
    },

    "inderpulate": {
        "kwargs": {
            "freq": random.randint(2,4),
            "lattice_drift": 1,
            "refract_range": random.randint(8,12),
            "ridges": True,
        }
    },

    "isoform": {
        "kwargs": {
            "invert": (random.random() < .5),
            "post_refract_range": .25 + random.random() * .25,
            "ridges": (random.random() < .5),
            "voronoi_alpha": .75 + random.random() * .25,
            "voronoi_func": 3,
            "with_outline":  3,
            "with_voronoi": random.randint(1,2),
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
            "distrib": [m.value for m in generators.ValueDistribution if m.name != "ones"][random.randint(0, len(generators.ValueDistribution) - 2)],
            "edges": .25 + random.random() * .5,
            "freq": [9, 12, 15, 18][random.randint(0,3)],
            "hsv_range": random.random() * .5,
            "octaves": random.randint(3,5),
            "point_distrib": [m.value for m in points.PointDistribution.grid_members()][random.randint(0, len(points.PointDistribution.grid_members()) - 1)],
            "point_freq": [3, 6, 9][random.randint(0,2)],
            "spline_order": 0,
            "voronoi_alpha": .25,
            "with_bloom": 0 if (random.random() < .5) else random.random(),
            "with_voronoi": 0 if (random.random() < .5) else 4,
        },

        "post_kwargs": {
            "with_dither": 0 if (random.random() < .5) else random.random() * .125,
        }
    },
     
    "misaligned": {
        "kwargs": {
            "freq": random.randint(12,24),
            "mask": [m.value for m in generators.ValueMask][random.randint(0, len(generators.ValueMask) - 1)],
            "octaves": random.randint(4,8),
            "spline_order": 0,
            "with_outline": 1,
        }
    },

    "muppet-skin": {
        "kwargs": {
            "freq": random.randint(2,3),
            "hsv_range": random.random() * .5,
            "lattice_drift": 0 if random.random() < .5 else random.random(),
            "with_bloom": .25 + random.random() * .5,
            "with_worms": 3 if random.random() < .5 else 1,
            "worms_bg": 0 if random.random() < .5 else random.random(),
            "worms_density": 500,
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

    "oldschool": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(2,5) * 2,
            "mask": "chess",
            "rgb": True,
            "spline_order": 0,
            "point_distrib": [m.value for m in points.PointDistribution][random.randint(0, len(points.PointDistribution) - 1)],
            "point_freq": random.randint(4,8),
            "voronoi_refract": random.randint(8,12),
            "with_voronoi": 6,
        }
    },
     
    "plaid": {
        "kwargs": {
            "deriv": 3,
            "distrib": "ones",
            "hsv_range": random.random() * .5,
            "freq": random.randint(3, 6) * 2,
            "mask": "chess",
            "octaves": random.randint(2, 5),
            "spline_order": random.randint(1, 3),
            "warp_freq": random.randint(2,3),
            "warp_range": random.random() * .25,
            "warp_octaves": 1,
        },

        "post_kwargs": {
            "with_dither": random.random() * 0.25,
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
            "hsv_saturation": random.random() * .5,
            "point_distrib": [m.value for m in points.PointDistribution.grid_members()][random.randint(0, len(points.PointDistribution.grid_members()) - 1)],
            "point_freq": random.randint(3, 8),
            "spline_order": 0,
            "voronoi_func": 2,
            "voronoi_nth": random.randint(0, 4),
            "voronoi_refract": 2,
            "with_bloom": random.random() * .5,
            "with_voronoi": 1,
        },

        "post_kwargs": {
            "with_dither": random.random() * .5,
        }
    },
     
    "reef": {
        "kwargs": {
            "freq": random.randint(3, 8),
            "hsv_range": random.random(),
            "lattice_drift": 1,
            "point_distrib": (["random"] + [m.value for m in points.PointDistribution.circular_members()])[random.randint(0, len(points.PointDistribution.circular_members()))],
            "point_freq": random.randint(2, 8),
            "ridges": True,
            "shadow": 1.0,
            "sin": random.random() * 2.5,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_refract": random.randint(8, 16),
            "with_bloom": 0.5,
            "with_voronoi": 6,
        }
    },
     
    "refractal": {
        "kwargs": {
            "invert": 1,
            "lattice_drift": 1,
            "point_freq": random.randint(8,10),
            "post_reflect_range": random.randint(96, 192),
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

    "soft-cells": {
        "kwargs": {
            "point_distrib": [m.value for m in points.PointDistribution][random.randint(0, len(points.PointDistribution) - 1)],
            "point_freq": random.randint(4,8),
            "voronoi_alpha": .5 + random.random() * .5,
            "with_bloom": .5 + random.random() * .5,
            "with_voronoi": 5,
        }
    },

    "soften": {
        "kwargs": {
            "freq": 2,
            "hsv_range": .25 + random.random() * .25,
            "hsv_rotation": random.random(),
            "lattice_drift": 1,
            "octaves": random.randint(1,4),
            "with_bloom": .25 + random.random() * .5,
        }
    },
     
    "spiral-in-spiral": {
        "kwargs": {
            "point_distrib": "spiral" if random.random() < .5 else "rotating",
            "point_freq": 10,
            "with_voronoi": random.randint(1,2),
            "with_worms": random.randint(1,4),
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": random.randint(5, 25),
        }
    },
     
    "spiraltown": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 1,
            "reflect_range": random.randint(3,6),
            "spline_order": random.randint(1,3),
            "with_wormhole": True,
            "wormhole_kink": random.randint(5,20),
            "wormhole_stride": random.random() * .05,
        }
    },
     
    "square-stripes": {
        "kwargs": {
            "hsv_range": random.random(),
            "point_distrib": [m.value for m in points.PointDistribution.grid_members()][random.randint(0, len(points.PointDistribution.grid_members()) - 1)],
            "point_freq": 2,
            "point_generations": random.randint(2,3),
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_func": random.randint(2,3),
            "voronoi_nth": random.randint(1,3),
            "voronoi_refract": 1.46,
            "with_voronoi": 2,
        }
    },
     
    "star-cloud": {
        "kwargs": {
            "deriv": 1,
            "freq": 2,
            "hsv_range": random.random() * 2.0,
            "invert": 1,
            "point_freq": 10,
            "reflect_range": random.random() + .5,
            "spline_order": 2,
            "voronoi_refract": random.randint(2,4),
            "with_bloom": .25 + random.random() * .5,
            "with_sobel": 1,
            "with_voronoi": 6,
        }
    },
     
    "stepper": {
        "kwargs": {
            "hsv_range": random.random(),
            "hsv_saturation": random.random(),
            "point_corners": (random.random() < .5),
            "point_distrib": [m.value for m in points.PointDistribution.circular_members()][random.randint(0, len(points.PointDistribution.circular_members()) - 1)],
            "point_freq": random.randint(7,10),
            "voronoi_func": random.randint(2,3),
            "voronoi_nth": random.randint(0, 48),
            "with_outline": 3,
            "with_voronoi": random.randint(1,5),
        }
    },

    "tensorflower": {
        "kwargs": {
            "corners": True,
            "hsv_range": random.random(),
            "freq": 2,
            "point_corners": True,
            "point_distrib": ["square", "h_hex", "v_hex"][random.randint(0,2)],
            "point_freq": 2,
            "spline_order": 0,
            "vortex_range": random.randint(8,25),
            "with_bloom": random.random(),
            "with_voronoi": 5,
        }
    },

    "traceroute": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(2,6),
            "mask": [m.value for m in generators.ValueMask][random.randint(0, len(generators.ValueMask) - 1)],
            "octaves": 8,
            "with_worms": random.randint(1,3),
            "worms_density": 500,
            "worms_kink": random.randint(5,25),
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
            "hsv_range": random.randint(0,3),
            "octaves": random.randint(1,2),
            "reflect_range": random.randint(6,8),
            "spline_order": random.randint(2,3),
            "with_wormhole": True,
            "wormhole_stride": random.random() * .0125,
        }
    },
     
    "warped-grid": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 12) * 2,
            "hsv_range": 3,
            "hsv_saturation": 0.27,
            "invert": 1,
            "mask": [m.value for m in generators.ValueMask][random.randint(0, len(generators.ValueMask) - 1)],
            "posterize_levels": 12,
            "spline_order": 0,
            "warp_interp": random.randint(1,3),
            "warp_freq": random.randint(2, 4),
            "warp_range": .25 + random.random() * .75,
            "warp_octaves": 1,
            "with_aberration": random.random() * .125,
            "with_bloom": random.randint(0, 1) * .5,
            "with_sobel": 2,
        }
    },

    "wireframe": {
        "kwargs": {
            "freq": random.randint(2, 5),
            "hsv_range": random.random(),
            "hsv_saturation": random.random(),
            "invert": 1,
            "lattice_drift": random.random(),
            "octaves": 2,
            "point_distrib": [m.value for m in points.PointDistribution.grid_members()][random.randint(0, len(points.PointDistribution.grid_members()) - 1)],
            "point_freq": random.randint(7,10),
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(1, 5),
            "warp_octaves": random.randint(1, 3),
            "warp_range": 0 if random.random() < .5 else random.random() * .5,
            "with_bloom": 0.25 + random.random() * .5,
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

    "woahdude-2": {
        "kwargs": {
            "freq": random.randint(2,3),
            "hsv_range": random.random() * 3.0,
            "shadow": random.random(),
            "sin": random.randint(5,15),
            "warp_range": random.randint(3,5),
            "warp_octaves": 3,
            "with_bloom": random.random(),
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
    Load a named preset.

    Returns a tuple of (dict, dict), containing `generators.multires` and `recipes.post_process` keyword args.

    See the `artmaker` script for an example of how these values are used.

    :param str preset_name: Name of the preset. If "random" is given, a random preset is returned.
    :return: Tuple(dict, dict)
    """

    if preset_name == "random":
        preset_name = list(PRESETS)[random.randint(0, len(PRESETS) - 1)]

        preset = PRESETS.get(preset_name)

    else:
        preset = PRESETS.get(preset_name, {})

    return preset.get("kwargs", {}), preset.get("post_kwargs", {}), preset_name