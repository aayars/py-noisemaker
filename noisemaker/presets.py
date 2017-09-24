""" Presets library for noisemaker/artmaker """

import random

from noisemaker.constants import PointDistribution, ValueDistribution, ValueMask

circular_dists = PointDistribution.circular_members()

grid_dists = PointDistribution.grid_members()


EFFECTS_PRESETS = {
    "be-kind-rewind": {
        "kwargs": {
            "with_aberration": random.random() * .02,
        },

        "post_kwargs": {
            "with_crt": True,
            "with_scan_error": True,
            "with_snow": .625 + random.random() * .125,
            "with_vhs": True,
        },
    },

    "corrupt": {
        "kwargs": {
            "warp_freq": [random.randint(4, 7), random.randint(1, 3)],
            "warp_octaves": random.randint(3, 5),
            "warp_range": .05 + random.random() * .45,
            "warp_interp": 0,
            "with_bloom": .5 + random.random() * .5,
        },

        "post_kwargs": {
            "with_glitch": random.random() > .125,
        }
    },

    "erosion-worms": {
        "kwargs": {
            "with_erosion_worms": True,
            "erosion_worms_alpha": .25 + random.random() * .75,
            "erosion_worms_contraction": .5 + random.random() * .5,
            "erosion_worms_density": random.randint(25, 100),
            "erosion_worms_iterations": random.randint(25, 100),
        }
    },

    "funhouse": {
        "kwargs": {
            "warp_freq": [random.randint(2, 4), random.randint(1, 4)],
            "warp_octaves": random.randint(1, 4),
            "warp_range": .25 + random.random() * .5,
        },
    },

    "glitchin-out": {
        "kwargs": {
            "with_aberration": random.random() * .02 if (random.random() > .333) else 0,
            "with_bloom": .5 + random.random() * .5 if (random.random() > .5) else 0,
        },

        "post_kwargs": {
            "with_crt": random.random() > .25,
            "with_dither": random.random() * .25 if random.random() > .5 else 0,
            "with_glitch": random.random() > .25,
            "with_scan_error": random.random() > .5,
        },
    },

    "light-leak": {
        "kwargs": {
            "vignette_brightness": random.randint(0, 1),
            "with_bloom": .25 + random.random() * .5,
            "with_light_leak": .5 + random.random() * .5,
            "with_vignette": .333 + random.random() * .333 if random.random() < .5 else None,
        },
    },

    "mosaic": {
        "kwargs": {
            "point_freq": 10,
            "voronoi_alpha": .75 + random.random() * .25,
            "with_voronoi": 5,
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "pop-art": {
        "kwargs": {
            "with_pop": True
        },
    },

    "reindex": {
        "kwargs": {
            "reindex_range": .125 + random.random() * .125,
        },
    },

    "reverb": {
        "kwargs": {
            "reverb_iterations": random.randint(1, 4),
            "with_reverb": random.randint(3, 6),
        },
    },

    "sobel-operator": {
        "kwargs": {
            "invert": random.random() > .5,
            "with_sobel": random.randint(1, 3),
        }
    },

    "voronoid": {
        "kwargs": {
            "point_freq": random.randint(4, 10),
            "voronoi_refract": random.random(),
            "voronoi_func": random.randint(1, 3),
            "with_voronoi": [1, 3, 6][random.randint(0, 2)]
        }
    },

    "vortex": {
        "kwargs": {
            "vortex_range": random.randint(12, 32),
        }
    },

    "wormhole-xtreme": {
        "kwargs": {
            "with_wormhole": True,
            "wormhole_stride": .025 + random.random() * .05,
            "wormhole_kink": .5 + random.random(),
        }
    },

    "worms": {
        "kwargs": {
            "with_worms": random.randint(1, 4),
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": 2.5,
            "worms_stride": 2.5,
            "worms_stride_deviation": 2.5,
        }
    },

}


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
            "voronoi_nth": 0 if random.random() < .5 else random.randint(0, 63),
            "with_voronoi": 2 if random.random() < .5 else random.randint(1, 5),
        }
    },

    "acid-grid": {
        "kwargs": {
            "invert": 1,
            "lattice_drift": 1,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
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

    "alien-terrain-multires": {
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

    "alien-terrain-worms": {
        "kwargs": {
            "deriv": 1,
            "deriv_alpha": 0.25 + random.random() * .25,
            "erosion_worms_alpha": .025 + random.random() * .0125,
            "erosion_worms_contraction": 2,
            "erosion_worms_density": random.randint(50, 75),
            "erosion_worms_iterations": random.randint(25, 50),
            "freq": random.randint(2, 6),
            "hsv_rotation": 0 if random.random() < .5 else random.random(),
            "hsv_saturation": 0.25 + random.random() * .25,
            "invert": random.randint(0, 1),
            "lattice_drift": 1,
            "octaves": 4,
            "point_freq": 10,
            "point_distrib": "random",
            "ridges": True,
            "shadow": .75,
            "sin": 1 + random.random(),
            "voronoi_alpha": 0.25 + random.random() * .25,
            "voronoi_refract": 0.25 + random.random() * .25,
            "with_bloom": 0.5 + random.random() * .25,
            "with_erosion_worms": True,
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_alpha": .05 + random.random() * .025,
            "worms_kink": random.randint(35, 50),
            "worms_duration": 0.4,
            "worms_density": 500,
        },

        "post_kwargs": {
            "with_dither": .1 + random.random() * .05,
        }
    },

    "aztec-waffles": {
        "kwargs": {
            "freq": 7,
            "invert": (random.random() < .5),
            "point_freq": random.randint(2, 4),
            "point_generations": 2,
            "point_distrib": "circular",
            "posterize_levels": random.randint(6, 18),
            "reflect_range": random.random() * 2,
            "spline_order": 0,
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(2, 4),
            "with_outline": 3,
            "with_voronoi": random.randint(1, 5),
        }
    },

    "bringing-hexy-back": {
        "kwargs": {
            "lattice_drift": 1,
            "point_distrib": "v_hex" if random.random() < .5 else "v_hex",
            "point_freq": 10,
            "post_deriv": 0 if random.random() < .5 else random.randint(1, 3),
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0 if random.random() < .5 else random.random(),
            "warp_octaves": 1,
            "warp_range": random.random() * .5,
            "with_bloom": 0.5,
            "with_voronoi": 5,
        }
    },

    "bubble-machine": {
        "kwargs": {
            "corners": True,
            "distrib": "uniform",
            "freq": random.randint(3, 6) * 2,
            "invert": random.randint(0, 1),
            "mask": ["h_hex", "v_hex"][random.randint(0, 1)],
            "posterize_levels": random.randint(8, 16),
            "reverb_iterations": random.randint(1, 3),
            "spline_order": random.randint(1, 3),
            "with_reverb": random.randint(3, 5),
            "with_outline": 1,
            "with_wormhole": True,
            "wormhole_kink": 1.0 + random.random() * 5,
            "wormhole_stride": .25 + random.random() * .75,
        }
    },

    "circulent": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "point_distrib": (["spiral"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_corners": True,
            "point_freq": random.randint(4, 8),
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_nth": random.randint(1, 4),
            "with_voronoi": random.randint(1, 2),
            "with_wormhole": True,
            "wormhole_kink": random.randint(3, 6),
            "wormhole_stride": .05 + random.random() * .05,
        }
    },

    "conjoined": {
        "kwargs": {
            "hsv_range": random.random(),
            "lattice_drift": 1,
            "point_distrib": ([m.value for m in circular_dists])[random.randint(0, len(circular_dists) - 1)],
            "point_freq": random.randint(3, 7),
            "post_deriv": 1,
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(0, 4),
            "voronoi_refract": random.randint(6, 12),
            "with_bloom": 0.25 + random.random() * .5,
            "with_voronoi": 2,
        }
    },

    "crop-spirals": {
        "kwargs": {
            "distrib": "laplace",
            "corners": False,
            "freq": 10,
            "hsv_range": 1,
            "hsv_rotation": .26,
            "hsv_saturation": .72,
            "mask": "v_hex",
            "reindex_range": .17,
            "spline_order": 2,
            "with_reverb": 2,
            "with_worms": 3,
            "worms_alpha": .95,
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": 2.21,
            "worms_stride": .55,
            "worms_stride_deviation": .06,
        }
    },

    "cubic": {
        "kwargs": {
            "freq": random.randint(2, 5),
            "point_distrib": "concentric",
            "point_freq": random.randint(3, 5),
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(2, 8),
            "with_bloom": 0.25 + random.random() * .5,
            "with_outline": 1,
            "with_voronoi": random.randint(1, 2),
        }
    },

    "deadlock": {
        "kwargs": {
           "hsv_range": random.random(),
           "hsv_rotation": random.random(),
           "hsv_saturation": random.random(),
           "point_corners": (random.random() < .5),
           "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
           "point_drift": 0 if random.random() < .5 else random.random(),
           "point_freq": 4,
           "point_generations": 2,
           "voronoi_func": random.randint(2, 3),
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
            "post_refract_range": random.randint(0, 1),
            "posterize_levels": random.randint(3, 5),
            "voronoi_alpha": 1,
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(1, 3),
            "with_aberration": random.random() * .01,
            "with_sobel": 2,
            "with_voronoi": random.randint(1, 4),
        },

        "post_kwargs": {
            "with_crt": True,
            "with_scan_error": True,
        }
    },

    "defocus": {
        "kwargs": {
            "freq": 12,
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "octaves": 5,
            "sin": 10,
            "with_bloom": 0.5,
        }
    },

    "dla-cells": {
        "kwargs": {
            "dla_padding": random.randint(2, 8),
            "hsv_range": random.random() * 1.5,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(2, 8),
            "voronoi_alpha": random.random(),
            "with_bloom": .25 + random.random() * .25,
            "with_dla": .5 + random.random() * .5,
            "with_voronoi": 0 if random.random() < .5 else random.randint(1, 5),
        }
    },

    "dla-forest": {
        "kwargs": {
            "dla_padding": random.randint(2, 8),
            "reverb_iterations": random.randint(2, 4),
            "with_bloom": .25 + random.random() * .25,
            "with_dla": 1,
            "with_reverb": random.randint(3, 6),
        }
    },

    "ears": {
        "kwargs": {
            "freq": 22,
            "distrib": "uniform",
            "hsv_range": random.random() * 2.5,
            "mask": [m.value for m in ValueMask if m.name != "chess"][random.randint(0, len(ValueMask) - 2)],
            "with_worms": 3,
            "worms_alpha": .875,
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
            "distrib": ["ones", "uniform"][random.randint(0, 1)],
            "freq": 12 * random.randint(1, 2),
            "hsv_range": random.random(),
            "invert": 1,
            "mask": [m.value for m in ValueMask if m.name != "chess"][random.randint(0, len(ValueMask) - 2)],
            "ridges": True,
            "shadow": 1,
            "spline_order": random.randint(2, 3),
            "with_outline": 1,
            "warp_freq": 2,
            "warp_octaves": 1,
            "warp_range": random.randint(1, 4),
        }
    },

    "fractile": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(2, 10),
            "reverb_iterations": random.randint(2, 4),
            "voronoi_nth": random.randint(0, 3),
            "with_reverb": random.randint(4, 8),
            "with_voronoi": random.randint(1, 5),
        }
    },

    "fuzzy-squares": {
        "kwargs": {
            "corners": True,
            "freq": 20,
            "distrib": "uniform",
            "mask": [m.value for m in ValueMask if m.name != "chess"][random.randint(0, len(ValueMask) - 2)],
            "spline_order": 1,
            "with_worms": 4,
            "worms_alpha": .5 + random.random() * .5,
            "worms_density": 400,
            "worms_duration": 2.0,
            "worms_stride": .5 + random.random(),
            "worms_stride_deviation": 1.0 + random.random(),
            "worms_kink": 1 + random.random() * 2.5,
        }
    },

    "fuzzy-swirls": {
        "kwargs": {
            "freq": random.randint(2, 32),
            "hsv_range": random.random() * 2,
            "point_freq": random.randint(8, 10),
            "spline_order": random.randint(1, 3),
            "voronoi_alpha": 0.5 * random.random() * .5,
            "with_voronoi": 6,
            "with_worms": random.randint(1, 4),
            "worms_density": 64,
            "worms_duration": 1,
            "worms_kink": 25,
        }
    },

    "fuzzy-thorns": {
        "kwargs": {
            "point_freq": random.randint(2, 4),
            "point_distrib": "waffle",
            "point_generations": 2,
            "voronoi_inverse": True,
            "voronoi_nth": random.randint(6, 12),
            "with_voronoi": 2,
            "with_worms": random.randint(1, 3),
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
            "reflect_range": random.randint(2, 4),
            "refract_range": random.randint(2, 4),
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
            "freq": random.randint(2, 6),
            "hsv_range": random.random(),
            "hsv_rotation": random.random(),
            "hsv_saturation": random.random(),
            "point_corners": True,
            "point_distrib": "concentric",
            "point_freq": random.randint(3, 6),
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
            "freq": random.randint(1, 3) * 2,
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
            "freq": random.randint(2, 4),
            "lattice_drift": 1,
            "refract_range": random.randint(8, 12),
            "ridges": True,
        }
    },

    "isoform": {
        "kwargs": {
            "hsv_range": random.random(),
            "invert": random.random() < .5,
            "post_deriv": 0 if random.random() < .5 else random.randint(1, 3),
            "post_refract_range": .25 + random.random() * .25,
            "ridges": random.random() < .5,
            "voronoi_alpha": .75 + random.random() * .25,
            "voronoi_func": random.randint(1, 3),
            "with_outline":  random.randint(1, 3),
            "with_voronoi": random.randint(1, 2),
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
            "distrib": [m.value for m in ValueDistribution if m.name != "ones"][random.randint(0, len(ValueDistribution) - 2)],
            "edges": .25 + random.random() * .5,
            "freq": [9, 12, 15, 18][random.randint(0, 3)],
            "hsv_range": random.random() * .5,
            "octaves": random.randint(3, 5),
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": [3, 6, 9][random.randint(0, 2)],
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
            "freq": random.randint(12, 24),
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "octaves": random.randint(4, 8),
            "spline_order": 0,
            "with_outline": 1,
        }
    },

    "muppet-skin": {
        "kwargs": {
            "freq": random.randint(2, 3),
            "hsv_range": random.random() * .5,
            "lattice_drift": 0 if random.random() < .5 else random.random(),
            "with_bloom": .25 + random.random() * .5,
            "with_worms": 3 if random.random() < .5 else 1,
            "worms_alpha": 0 if random.random() < .5 else (.75 + random.random() * .25),
            "worms_density": 500,
        }
    },

    "neon-cambrian": {
        "kwargs": {
            "hsv_range": 1,
            "invert": 1,
            "posterize_levels": 24,
            "with_aberration": 0.01,
            "with_bloom": .25 + random.random() * .5,
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
            "freq": random.randint(2, 5),
            "lattice_drift": random.randint(0, 1),
            "octaves": random.randint(3, 6),
            "ridges": True,
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

    "octave-rings": {
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

    "oldschool": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(2, 5) * 2,
            "mask": "chess",
            "rgb": True,
            "spline_order": 0,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(4, 8),
            "voronoi_refract": random.randint(8, 12),
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
            "warp_freq": random.randint(2, 3),
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
            "with_bloom": .5 + random.random() * .5,
            "with_outline": 1,
        },

        "post_kwargs": {
            "with_dither": 0.25,
        }
    },

    "quilty": {
        "kwargs": {
            "freq": random.randint(2, 6),
            "hsv_saturation": random.random() * .5,
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": random.randint(3, 8),
            "spline_order": 0,
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(0, 4),
            "voronoi_refract": random.randint(1, 3),
            "with_bloom": .25 + random.random() * .5,
            "with_voronoi": random.randint(1, 2),
        },

        "post_kwargs": {
            "with_dither": random.random() * .5,
        }
    },

    "redmond": {
        "kwargs": {
            "corners": True,
            "distrib": "uniform",
            "freq": 8,
            "hsv_range": random.random() * 4.0,
            "invert": random.randint(0, 1),
            "mask": "square",
            "point_generations": 2,
            "point_freq": 2,
            "point_distrib": ["chess", "square"][random.randint(0, 1)],
            "point_corners": True,
            "reverb_iterations": random.randint(1, 3),
            "spline_order": 0,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_inverse": random.randint(0, 1),
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(0, 3),
            "with_bloom": .25 + random.random() * .5,
            "with_reverb": random.randint(3, 6),
            "with_voronoi": random.randint(1, 6),
        },

        "post_kwargs": {
            "with_dither": 0.13,
            "with_snow": 0.25,
        }
    },

    "reef": {
        "kwargs": {
            "freq": random.randint(3, 8),
            "hsv_range": random.random(),
            "lattice_drift": 1,
            "point_distrib": (["random"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_freq": random.randint(2, 8),
            "ridges": True,
            "shadow": 1.0,
            "sin": random.random() * 2.5,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_refract": random.randint(8, 16),
            "with_bloom": .25 + random.random() * .5,
            "with_voronoi": 6,
        }
    },

    "refractal": {
        "kwargs": {
            "invert": 1,
            "lattice_drift": 1,
            "octaves": random.randint(1, 3),
            "point_freq": random.randint(8, 10),
            "post_reflect_range": random.randint(96, 192),
            "sin": random.random() * 10.0,
            "voronoi_alpha": .5 + random.random() * .5,
            "with_voronoi": 6,
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
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(4, 8),
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
            "octaves": random.randint(1, 4),
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "spiral-in-spiral": {
        "kwargs": {
            "point_distrib": "spiral" if random.random() < .5 else "rotating",
            "point_freq": 10,
            "with_voronoi": random.randint(1, 2),
            "with_worms": random.randint(1, 4),
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": random.randint(5, 25),
        }
    },

    "spiraltown": {
        "kwargs": {
            "freq": 2,
            "hsv_range": 1,
            "reflect_range": random.randint(3, 6),
            "spline_order": random.randint(1, 3),
            "with_wormhole": True,
            "wormhole_kink": random.randint(5, 20),
            "wormhole_stride": random.random() * .05,
        }
    },

    "square-stripes": {
        "kwargs": {
            "hsv_range": random.random(),
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": 2,
            "point_generations": random.randint(2, 3),
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(1, 3),
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
            "voronoi_refract": random.randint(2, 4),
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
            "point_distrib": [m.value for m in circular_dists][random.randint(0, len(circular_dists) - 1)],
            "point_freq": random.randint(7, 10),
            "voronoi_func": random.randint(2, 3),
            "voronoi_nth": random.randint(0, 48),
            "with_outline": 3,
            "with_voronoi": random.randint(1, 5),
        }
    },

    "tensorflower": {
        "kwargs": {
            "corners": True,
            "hsv_range": random.random(),
            "freq": 2,
            "point_corners": True,
            "point_distrib": ["square", "h_hex", "v_hex"][random.randint(0, 2)],
            "point_freq": 2,
            "spline_order": 0,
            "vortex_range": random.randint(8, 25),
            "with_bloom": .25 + random.random() * .5,
            "with_voronoi": 5,
        }
    },

    "the-data-must-flow": {
        "kwargs": {
            "freq": 2,
            "hsv_range": random.random() * 2.5,
            "invert": 1,
            "with_bloom": .25 + random.random() * .5,
            "with_sobel": 1,
            "with_worms": 1,
            "worms_alpha": .9 + random.random() * .1,
            "worms_density": 1.5 + random.random(),
            "worms_duration": 1,
            "worms_stride": 8,
            "worms_stride_deviation": 6,
         }
    },

    "traceroute": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(2, 6),
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "octaves": 8,
            "with_worms": random.randint(1, 3),
            "worms_density": 500,
            "worms_kink": random.randint(5, 25),
        }
    },

    "triangular": {
        "kwargs": {
            "corners": True,
            "distrib": ["ones", "uniform"][random.randint(0, 1)],
            "freq": random.randint(1, 4) * 2,
            "invert": random.randint(0, 1),
            "mask": ["h_tri", "v_tri"][random.randint(0, 1)],
            "octaves": random.randint(4, 8),
            "with_sobel": random.randint(0, 1),
        }
    },

    "tribbles": {
        "kwargs": {
            "freq": random.randint(2, 8),
            "hsv_rotation": 0.375 + random.random() * .15,
            "hsv_saturation": .375 + random.random() * .15,
            "invert": 1,
            "octaves": 3,
            "point_distrib": "h_hex",
            "point_freq": random.randint(3, 4) * 2,
            "ridges": True,
            "voronoi_alpha": 0.5 + random.random() * .25,
            "warp_octaves": random.randint(4, 6),
            "warp_range": 0.05 + random.random() * .05,
            "with_bloom": 0.25 + random.random() * .5,
            "with_voronoi": 5,
            "with_worms": 3,
            "worms_alpha": .75 + random.random() * .25,
            "worms_density": 750,
            "worms_duration": .5,
            "worms_stride_deviation": .5,
        }
    },

    "velcro": {
        "kwargs": {
            "freq": 2,
            "hsv_range": random.randint(0, 3),
            "octaves": random.randint(1, 2),
            "reflect_range": random.randint(6, 8),
            "spline_order": random.randint(2, 3),
            "with_wormhole": True,
            "wormhole_stride": random.random() * .0125,
        }
    },

    "vortex-checkers": {
        "kwargs": {
            "freq": random.randint(4, 10) * 2,
            "distrib": ["ones", "uniform", "laplace"][random.randint(0, 2)],
            "mask": "chess",
            "hsv_range": random.random(),
            "hsv_saturation": random.random(),
            "outline": 3,
            "posterize": random.randint(10, 15),
            "reverb_iterations": random.randint(2, 4),
            "sin": .5 + random.random(),
            "spline_order": 0,
            "vortex_range": 2.5 + random.random() * 5,
            "with_reverb": random.randint(3, 5),
        }
    },

    "warped-grid": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 48) * 2,
            "hsv_range": 3,
            "hsv_saturation": 0.27,
            "invert": 1,
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "posterize_levels": 12,
            "spline_order": 0,
            "warp_interp": random.randint(1, 3),
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
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": random.randint(7, 10),
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
            "point_distrib": (["spiral"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_freq": 10,
            "shadow": .5,
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_refract": random.randint(1, 3),
            "with_bloom": 0.25 + random.random() * .5,
            "with_voronoi": random.randint(1, 2),
         }
     },

    "woahdude-voronoi-refract": {
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

    "woahdude-octave-warp": {
        "kwargs": {
            "freq": random.randint(2, 3),
            "hsv_range": random.random() * 3.0,
            "shadow": random.random(),
            "sin": random.randint(5, 15),
            "warp_range": random.randint(3, 5),
            "warp_octaves": 3,
            "with_bloom": .25 + random.random() * .5,
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
            "worms_alpha": .875,
            "worms_density": 346.75,
            "worms_duration": 2.2,
            "worms_kink": 6.47,
            "worms_stride": 2.5,
            "worms_stride_deviation": 1.25,
        }
    },

    "wormstep": {
        "kwargs": {
            "corners": True,
            "freq": random.randint(2, 4),
            "lattice_drift": 0 if random.random() < .5 else 1.0,
            "octaves": random.randint(1, 3),
            "with_bloom": .25 + random.random() * .25,
            "with_worms": 4,
            "worms_alpha": .5 + random.random() * .5,
            "worms_density": 500,
            "worms_kink": 1.0 + random.random() * 4.0,
            "worms_stride": 8.0 + random.random() * 4.0,
        }
    }
}


def load(preset_name, preset_set=None):
    """
    Load a named preset.

    Returns a tuple of (dict, dict), containing `generators.multires` and `recipes.post_process` keyword args.

    See the `artmaker` script for an example of how these values are used.

    :param str preset_name: Name of the preset. If "random" is given, a random preset is returned.
    :return: Tuple(dict, dict)
    """

    if preset_set is None:
        preset_set = PRESETS

    if preset_name == "random":
        preset_name = list(preset_set)[random.randint(0, len(preset_set) - 1)]

        preset = preset_set.get(preset_name)

    else:
        preset = preset_set.get(preset_name, {})

    return preset.get("kwargs", {}), preset.get("post_kwargs", {}), preset_name
