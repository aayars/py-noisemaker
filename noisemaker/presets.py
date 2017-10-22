""" Presets library for artmaker/artmangler scripts """

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
            "erosion_worms_alpha": .25 + random.random() * .75,
            "erosion_worms_contraction": .5 + random.random() * .5,
            "erosion_worms_density": random.randint(25, 100),
            "erosion_worms_iterations": random.randint(25, 100),
            "with_erosion_worms": True,
        }
    },

    "extract-derivative": {
        "kwargs": {
            "deriv": random.randint(1, 3),
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
            "with_aberration": random.random() * .02 if random.random() > .333 else 0,
            "with_bloom": .5 + random.random() * .5 if random.randint(0, 1) else 0,
        },

        "post_kwargs": {
            "with_crt": random.random() > .25,
            "with_dither": random.random() * .25 if random.randint(0, 1) else 0,
            "with_glitch": True,
            "with_scan_error": random.randint(0, 1),
        },
    },

    "light-leak": {
        "kwargs": {
            "vignette_brightness": random.randint(0, 1),
            "with_bloom": .25 + random.random() * .5,
            "with_light_leak": .5 + random.random() * .5,
            "with_vignette": .333 + random.random() * .333 if random.randint(0, 1) else None,
        },
    },

    "mosaic": {
        "kwargs": {
            "point_distrib": "random" if random.randint(0, 1) else [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": 10,
            "voronoi_alpha": .75 + random.random() * .25,
            "with_voronoi": 5,
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "needs-more-jpeg": {
        "kwargs": {
            "with_jpeg_decimate": random.randint(10, 25),
        },
    },

    "pop-art": {
        "kwargs": {
            "with_pop": True
        },
    },

    "posterize-outline": {
        "kwargs": {
            "posterize_levels": random.randint(3, 7),
            "with_outline": 1,
        }
    },

    "reflect-domain-warp": {
        "kwargs": {
            "reflect_range": .125 + random.random() * 2.5,
        },
    },

    "refract-domain-warp": {
        "kwargs": {
            "refract_range": .125 + random.random() * 2.5,
        },
    },

    "reindex": {
        "kwargs": {
            "reindex_range": .125 + random.random() * 2.5,
        },
    },

    "reverb": {
        "kwargs": {
            "reverb_iterations": random.randint(1, 4),
            "with_reverb": random.randint(3, 6),
        },
    },

    "shadows": {
        "kwargs": {
            "with_shadow": .5 + random.random() * .5,
            "with_vignette": .5 + random.random() * .5,
            "vignette_brightness": 0,
        }
    },

    "sobel-operator": {
        "kwargs": {
            "invert": random.randint(0, 1),
            "with_sobel": random.randint(1, 3),
        }
    },

    "swerve-h": {
        "kwargs": {
            "warp_freq": [random.randint(3, 6), 1],
            "warp_octaves": 1,
            "warp_range": 1.0 + random.random(),
        },
    },

    "swerve-v": {
        "kwargs": {
            "warp_freq": [1, random.randint(3, 6)],
            "warp_octaves": 1,
            "warp_range": 1.0 + random.random(),
        },
    },

    "voronoid": {
        "kwargs": {
            "point_freq": random.randint(4, 10),
            "voronoi_func": random.randint(1, 3),
            "voronoi_refract": .5 + random.random() * .5,
            "voronoi_nth": random.randint(0, 3),
            "with_voronoi": [1, 3, 6][random.randint(0, 2)]
        }
    },

    "vortex": {
        "kwargs": {
            "vortex_range": random.randint(8, 24),
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
            "worms_alpha": .75 + random.random() * .25,
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
            "voronoi_nth": 0 if random.randint(0, 1) else random.randint(0, 63),
            "with_voronoi": 2 if random.randint(0, 1) else random.randint(1, 5),
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
            "deriv_alpha": .333 + random.random() * .333,
            "freq": random.randint(4, 6),
            "saturation": .25 + random.random() * .25,
            "invert": 1,
            "lattice_drift": 1,
            "octaves": 8,
            "with_bloom": .5 + random.random() * .25,
            "with_shadow": .5 + random.random() * .375,
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
            "hue_rotation": 0 if random.randint(0, 1) else random.random(),
            "saturation": 0.25 + random.random() * .25,
            "invert": random.randint(0, 1),
            "lattice_drift": 1,
            "octaves": 4,
            "point_freq": 10,
            "point_distrib": "random",
            "ridges": True,
            "sin": 1 + random.random(),
            "voronoi_alpha": 0.333 + random.random() * .333,
            "voronoi_refract": 0.333 + random.random() * .333,
            "with_bloom": 0.5 + random.random() * .25,
            "with_erosion_worms": True,
            "with_shadow": .5 + random.random() * .375,
            "with_voronoi": 6,
            "with_worms": 4,
            "worms_alpha": .1 + random.random() * .025,
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
            "invert": random.randint(0, 1),
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

    "berkeley": {
        "kwargs": {
            "freq": random.randint(12, 16),
            "octaves": 8,
            "post_ridges": True,
            "reindex_range": .375 + random.random() * .125,
            "ridges": True,
            "sin": 2 * random.random() * 2,
            "with_shadow": 1,
        }
    },

    "blacklight-fantasy": {
        "kwargs": {
            "invert": 1,
            "post_hue_rotation": -.125,
            "posterize_levels": 3,
            "rgb": True,
            "voronoi_func": random.randint(1, 3),
            "voronoi_nth": random.randint(0, 3),
            "voronoi_refract": 1.0 + random.random() * 2.5,
            "warp_octaves": random.randint(1, 4),
            "warp_range": None if random.randint(0, 1) else random.random() * 2.0,
            "with_bloom": .5 + random.random() * .5,
            "with_sobel": 2,
            "with_voronoi": random.randint(1, 6),
        },

        "post_kwargs": {
            "with_dither": .075 + random.random() * .075,
        },
    },

    "blobby": {
        "kwargs": {
            "deriv": random.randint(1, 3),
            "distrib": "uniform",
            "freq": random.randint(6, 12) * 2,
            "saturation": .25 + random.random() * .5,
            "hue_range": .25 + random.random() * .5,
            "hue_rotation": None if random.randint(0, 1) else random.random(),
            "invert": 1,
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "outline": 1,
            # "posterize_levels": random.randint(10, 20),
            "reverb_iterations": random.randint(2, 4),
            "spline_order": random.randint(2, 3),
            "warp_freq": random.randint(6, 12),
            "warp_interp": random.randint(1, 3),
            "warp_octaves": random.randint(2, 4),
            "warp_range": .05 + random.random() * .1,
            "with_reverb": random.randint(1, 3),
            "with_shadow": 1,
        }
    },

    "bringing-hexy-back": {
        "kwargs": {
            "lattice_drift": 1,
            "point_distrib": "v_hex" if random.randint(0, 1) else "v_hex",
            "point_freq": 10,
            "post_deriv": 0 if random.randint(0, 1) else random.randint(1, 3),
            "voronoi_alpha": 0.5,
            "voronoi_refract": 0 if random.randint(0, 1) else random.random(),
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

    "cell-refract": {
        "kwargs": {
            "with_voronoi": 1,
            "point_freq": random.randint(3, 4),
            "post_ridges": True,
            "reindex_range": 1.0 + random.random() * 1.5,
            "voronoi_refract": random.randint(8, 12),
            "ridges": True,
        }
    },

    "circulent": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "invert": 1,
            "point_distrib": (["spiral"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_corners": True,
            "point_freq": random.randint(4, 8),
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_nth": random.randint(1, 4),
            "with_reverb": random.randint(1, 3),
            "with_voronoi": random.randint(1, 4),
            "with_wormhole": True,
            "wormhole_kink": random.randint(3, 6),
            "wormhole_stride": .05 + random.random() * .05,
        }
    },

    "conjoined": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "hue_range": random.random(),
            "point_distrib": ([m.value for m in circular_dists])[random.randint(0, len(circular_dists) - 1)],
            "point_freq": random.randint(3, 7),
            "post_deriv": 1,
            "reindex_range": .5 + random.random() * .5,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_nth": random.randint(0, 4),
            "voronoi_refract": random.randint(2, 4),
            "with_bloom": .25 + random.random() * .5,
            "with_reverb": 2,
            "with_voronoi": 2,
        }
    },

    "crop-spirals": {
        "kwargs": {
            "distrib": "laplace",
            "corners": False,
            "freq": random.randint(4, 6) * 2,
            "hue_range": 1,
            "saturation": .75,
            "mask": ["h_hex", "v_hex"][random.randint(0, 1)],
            "reindex_range": .1 + random.random() * .1,
            "spline_order": 2,
            "with_reverb": random.randint(2, 4),
            "with_worms": 3,
            "worms_alpha": .9 + random.random() * .1,
            "worms_density": 500,
            "worms_duration": 1,
            "worms_kink": 2 + random.random(),
            "worms_stride": .333 + random.random() * .333,
            "worms_stride_deviation": .04 + random.random() * .04,
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
           "hue_range": random.random(),
           "hue_rotation": random.random(),
           "saturation": random.random(),
           "point_corners": random.randint(0, 1),
           "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
           "point_drift": 0 if random.randint(0, 1) else random.random(),
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
            "with_voronoi": 1,
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
            "hue_range": random.random() * 1.5,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(2, 8),
            "voronoi_alpha": random.random(),
            "with_bloom": .25 + random.random() * .25,
            "with_dla": .5 + random.random() * .5,
            "with_voronoi": 0 if random.randint(0, 1) else random.randint(1, 5),
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
            "hue_range": random.random() * 2.5,
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
            "hue_range": random.random(),
            "invert": 1,
            "mask": [m.value for m in ValueMask if m.name != "chess"][random.randint(0, len(ValueMask) - 2)],
            "ridges": True,
            "spline_order": random.randint(2, 3),
            "with_outline": 1,
            "warp_freq": 2,
            "warp_octaves": 1,
            "warp_range": random.randint(1, 4),
            "with_shadow": 1,
        }
    },

    "flowbie": {
        "kwargs": {
            "freq": random.randint(2, 4),
            "octaves": random.randint(1, 2),
            # "post_deriv": random.randint(1, 3),
            "with_bloom": .25 + random.random() * .5,
            "with_wormhole": True,
            "with_worms": random.randint(1, 3),
            "refract": random.randint(0, 3),
            "wormhole_alpha": .333 + random.random() * .333,
            "wormhole_kink": .25 + random.random() * .25,
            "wormhole_stride": random.random() * 2.5,
            "worms_alpha": .125 + random.random() * .125,
            "worms_stride": .25 + random.random() * .25,
        }
    },

    "fractile": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": random.randint(2, 10),
            "reverb_iterations": random.randint(2, 4),
            "voronoi_alpha": min(.75 + random.random() * .5, 1),
            "voronoi_func": random.randint(1, 3),
            "voronoi_nth": random.randint(0, 3),
            "with_bloom": .25 + random.random() * .5,
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
            "hue_range": random.random() * 2,
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

    "game-show": {
        "kwargs": {
            "distrib": "normal",
            "freq": random.randint(8, 16) * 2,
            "mask": ["h_tri", "v_tri"][random.randint(0, 1)],
            "posterize_levels": random.randint(2, 5),
            "spline_order": 2,
            "with_snow": random.random() * .333,
        },

        "post_kwargs": {
            "with_crt": True,
        }
    },

    "glass-onion": {
        "kwargs": {
            "point_freq": random.randint(3, 6),
            "post_deriv": random.randint(1, 3),
            "post_refract_range": .75 + random.random() * .5,
            "voronoi_inverse": random.randint(0, 1),
            "with_reverb": random.randint(3, 5),
            "with_voronoi": 2,
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
            "hue_range": 0,
            "hue_rotation": random.random(),
            "saturation": 0.27,
            "invert": random.randint(0, 1),
            "mask": "chess",
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

    "gravy": {
        "kwargs": {
            "distrib": "ones",
            "freq": 24 * random.randint(2, 6),
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "post_deriv": 2,
            "spline_order": random.randint(1, 2),
            "warp_range": .25 + random.random() * .5,
            "warp_octaves": 3,
            "warp_freq": random.randint(2, 4),
            "warp_interp": 3,
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "hairy-diamond": {
        "kwargs": {
            "erosion_worms_alpha": .75 + random.random() * .25,
            "erosion_worms_contraction": .5 + random.random(),
            "erosion_worms_density": random.randint(25, 50),
            "erosion_worms_iterations": random.randint(25, 50),
            "freq": random.randint(2, 6),
            "hue_range": random.random(),
            "hue_rotation": random.random(),
            "saturation": random.random(),
            "point_corners": True,
            "point_distrib": [m.value for m in circular_dists][random.randint(0, len(circular_dists) - 1)],
            "point_freq": random.randint(3, 6),
            "point_generations": 2,
            "spline_order": random.randint(0, 3),
            "voronoi_func": random.randint(2, 3),
            "voronoi_inverse": True,
            "voronoi_alpha": .25 + random.random() * .5,
            "with_erosion_worms": True,
            "with_voronoi": [1, 6][random.randint(0, 1)],
        }
    },

    "halt-catch-fire": {
        "kwargs": {
            "freq": 2,
            "hue_range": .05,
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
            "hue_range": .125 + random.random() * 2.0,
            "lattice_drift": random.random(),
        }
    },

    "hsv-shadows": {
        "kwargs": {
            "brightness_distrib": "mids",
            "distrib": "uniform",
            "freq": random.randint(12, 24),
            "hue_range": .25 + random.random() * .25,
            "saturation": .25 + random.random() * .25,
            "saturation_distrib": "uniform",
            "with_shadow": 1,
        }
    },

    "invaders": {
        "kwargs": {
             "distrib": "ones",
             "freq": 24 * random.randint(1, 8),
             "mask": [m.value for m in ValueMask.procedural_members()][random.randint(0, len(ValueMask.procedural_members()) - 1)],
             "spline_order": 0,
             "with_bloom": .25 + random.random() * .25,
        },

        "post_kwargs": {
             "with_crt": True,
             "with_scan_error": True,
        }
    },

    "isoform": {
        "kwargs": {
            "hue_range": random.random(),
            "invert": random.randint(0, 1),
            "post_deriv": 0 if random.randint(0, 1) else random.randint(1, 3),
            "post_refract_range": .25 + random.random() * .25,
            "ridges": random.randint(0, 1),
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
            "hue_range": random.random() * .5,
            "octaves": random.randint(3, 5),
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": [3, 6, 9][random.randint(0, 2)],
            "spline_order": 0,
            "voronoi_alpha": .25,
            "with_bloom": 0 if random.randint(0, 1) else random.random(),
            "with_voronoi": 0 if random.randint(0, 1) else 4,
        },

        "post_kwargs": {
            "with_dither": 0 if random.randint(0, 1) else random.random() * .125,
        }
    },

    "misaligned": {
        "kwargs": {
            "distrib": [m.value for m in ValueDistribution][random.randint(0, len(ValueDistribution) - 1)],
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
            "hue_range": random.random() * .5,
            "lattice_drift": 0 if random.randint(0, 1) else random.random(),
            "with_bloom": .25 + random.random() * .5,
            "with_worms": 3 if random.randint(0, 1) else 1,
            "worms_alpha": .75 + random.random() * .25,
            "worms_density": 500,
        }
    },

    "neon-cambrian": {
        "kwargs": {
            "hue_range": 1,
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
            "freq": random.randint(3, 10),
            "hue_range": random.random(),
            "saturation": .5 + random.random() * .5,
            "lattice_drift": random.randint(0, 1),
            "octaves": random.randint(2, 4),
            "reverb_iterations": random.randint(1, 2),
            "point_freq": random.randint(3, 10),
            "spline_order": 0,
            "voronoi_refract": random.randint(1, 4),
            "warp_freq": random.randint(2, 4),
            "warp_interp": 3,
            "warp_octaves": 1,
            "warp_range": .075 + random.random() * .075,
            "with_outline": 1,
            "with_reverb": random.randint(0, 1),
            "with_voronoi": 6,
        }
    },

    "octave-rings": {
        "kwargs": {
            "corners": True,
            "distrib": "ones",
            "freq": random.randint(4, 10),
            "hue_range": random.random(),
            "saturation": 0.5,
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
            "hue_range": random.random() * .5,
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

    "pluto": {
        "kwargs": {
            "freq": random.randint(4, 8),
            "deriv": random.randint(1, 3),
            "deriv_alpha": .333 + random.random() * .333,
            "hue_rotation": .575,
            "octaves": 10,
            "point_freq": random.randint(6, 8),
            "refract_range": .075 + random.random() * .075,
            "ridges": True,
            "saturation": .125 + random.random() * .075,
            "voronoi_alpha": .75,
            "voronoi_nth": random.randint(1, 3),
            "with_bloom": .25 + random.random() * .25,
            "with_shadow": .75 + random.random() * .25,
            "with_voronoi": 2,
        },

        "post_kwargs": {
            "with_dither": .075 + random.random() * .075,
        }
    },

    "political-map": {
        "kwargs": {
            "freq": 5,
            "saturation": 0.35,
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

    "prophesy": {
        "kwargs": {
            "distrib": "ones",
            "emboss": .5 + random.random() * .5,
            "freq": 48,
            "saturation": .075 + random.random() * .075,
            "invert": random.randint(0, 1),
            "mask": "invaders",
            "octaves": 2,
            "refract_range": .5,
            "spline_order": random.randint(1, 2),
            "posterize_levels": random.randint(4, 8),
        }
    },

    "quilty": {
        "kwargs": {
            "freq": random.randint(2, 6),
            "saturation": random.random() * .5,
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
            "hue_range": random.random() * 4.0,
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
            "hue_range": random.random(),
            "lattice_drift": 1,
            "point_distrib": (["random"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_freq": random.randint(2, 8),
            "ridges": True,
            "sin": random.random() * 2.5,
            "voronoi_alpha": .5 + random.random() * .5,
            "voronoi_refract": random.randint(8, 16),
            "with_bloom": .25 + random.random() * .5,
            "with_shadow": 1.0,
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

    "rgb-shadows": {
        "kwargs": {
            "brightness_distrib": "mids",
            "distrib": "uniform",
            "freq": random.randint(6, 16),
            "hue_range": random.randint(1, 4),
            "lattice_drift": random.random(),
            "saturation_distrib": "ones",
            "with_shadow": 1,
        }
    },

    "ridged-bubbles": {
        "kwargs": {
            "corners": True,
            "freq": 2,
            "invert": True,
            "point_distrib": [m.value for m in PointDistribution][random.randint(0, len(PointDistribution) - 1)],
            "point_freq": random.randint(4, 10),
            "post_ridges": True,
            "voronoi_alpha": .25 + random.random() * .25,
            "with_reverb": random.randint(0, 2),
            "with_voronoi": 2,
        }
    },

    "ridged-ridges": {
        "kwargs": {
            "freq": random.randint(2, 8),
            "lattice-drift": random.randint(0, 1),
            "octaves": random.randint(3, 6),
            "post_ridges": True,
            "ridges": True,
        }
    },

    "shatter": {
        "kwargs": {
             "freq": random.randint(2, 4),
             "invert": random.randint(0, 1),
             "point_freq": random.randint(3, 6),
             "post_refract_range": random.randint(3, 5),
             "posterize_levels": random.randint(4, 6),
             "voronoi_func": [1, 3][random.randint(0, 1)],
             "voronoi_inverse": random.randint(0, 1),
             "with_outline": random.randint(1, 3),
             "with_voronoi": 5,
        }
    },

    "shmoo": {
        "kwargs": {
            "freq": random.randint(4, 6),
            "hue_range": 2 + random.random(),
            "invert": 1,
            "posterize_levels": random.randint(3, 5),
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
            "hue_range": .25 + random.random() * .25,
            "hue_rotation": random.random(),
            "lattice_drift": 1,
            "octaves": random.randint(1, 4),
            "with_bloom": .25 + random.random() * .5,
        }
    },

    "spiral-clouds": {
        "kwargs": {
            "freq": random.randint(2, 4),
            "lattice_drift": 1.0,
            "octaves": random.randint(4, 8),
            "saturation-distrib": "ones",
            "shadow": 1,
            "with_wormhole": True,
            "wormhole_alpha": .333 + random.random() * .333,
            "wormhole_stride": .001 + random.random() * .0005,
            "wormhole_kink": random.randint(40, 50),
        }
    },

    "spiral-in-spiral": {
        "kwargs": {
            "point_distrib": "spiral" if random.randint(0, 1) else "rotating",
            "point_freq": 10,
            "reverb_iterations": random.randint(1, 4),
            "with_reverb": random.randint(0, 6),
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
            "hue_range": 1,
            "reflect_range": random.randint(3, 6),
            "spline_order": random.randint(1, 3),
            "with_wormhole": True,
            "wormhole_kink": random.randint(5, 20),
            "wormhole_stride": random.random() * .05,
        }
    },

    "square-stripes": {
        "kwargs": {
            "hue_range": random.random(),
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
            "hue_range": random.random() * 2.0,
            "invert": 1,
            "point_freq": 10,
            "reflect_range": random.random() + .5,
            "reverb_iterations": random.randint(1, 4),
            "spline_order": 2,
            "voronoi_refract": random.randint(2, 4),
            "with_bloom": .25 + random.random() * .5,
            "with_reverb": random.randint(0, 3),
            "with_sobel": 1,
            "with_voronoi": 6,
        }
    },

    "stepper": {
        "kwargs": {
            "hue_range": random.random(),
            "saturation": random.random(),
            "point_corners": random.randint(0, 1),
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
            "hue_range": random.random(),
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
            "hue_range": random.random() * 2.5,
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
            "hue_rotation": 0.375 + random.random() * .15,
            "saturation": .375 + random.random() * .15,
            "invert": 1,
            "octaves": 3,
            "point_distrib": "h_hex",
            "point_freq": random.randint(3, 4) * 2,
            "ridges": True,
            "voronoi_alpha": 0.5 + random.random() * .25,
            "warp_freq": random.randint(2, 4),
            "warp_octaves": random.randint(2, 4),
            "warp_range": 0.05 + random.random() * .01,
            "with_bloom": 0.25 + random.random() * .5,
            "with_voronoi": 5,
            "with_worms": 3,
            "worms_alpha": .75 + random.random() * .25,
            "worms_density": 750,
            "worms_duration": .5,
            "worms_stride_deviation": .5,
        }
    },

    "triblets": {
        "kwargs": {
            "distrib": "uniform",
            "freq": random.randint(3, 15) * 2,
            "mask": [m.value for m in ValueMask][random.randint(0, len(ValueMask) - 1)],
            "hue_rotation": 0.875 + random.random() * .15,
            "saturation": .375 + random.random() * .15,
            "octaves": random.randint(3, 6),
            "warp_octaves": random.randint(1, 2),
            "warp_freq": random.randint(2, 4),
            "warp_range": 0.05 + random.random() * .1,
            "with_bloom": 0.25 + random.random() * .5,
            "with_worms": 3,
            "worms_alpha": .75 + random.random() * .25,
            "worms_density": 750,
            "worms_duration": .5,
            "worms_stride": .5,
            "worms_stride_deviation": .25,
        }
    },

    "unicorn-puddle": {
        "kwargs": {
            "emboss": 1,
            "freq": random.randint(3, 5),
            "hue_range": 1.0 + random.random() * .5,
            "invert": .5 * random.random() * .5,
            "lattice_drift": 1,
            "octaves": random.randint(4, 6),
            "with_bloom": .5 + random.random() * .5,
        }
    },

    "vectoroids": {
        "kwargs": {
            "freq": 25,
            "distrib": "ones",
            "mask": "sparse",
            "point_freq": 10,
            "point_drift": .25 + random.random() * .75,
            "post_deriv": 1,
            "spline_order": 0,
            "with_aberration": .0025 + random.random() * .0075,
            "with_voronoi": 4,
        },
    },

    "velcro": {
        "kwargs": {
            "freq": 2,
            "hue_range": random.randint(0, 3),
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
            "hue_range": random.random(),
            "saturation": random.random(),
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
            "hue_range": 3,
            "saturation": 0.27,
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
            "hue_range": random.random(),
            "saturation": random.random(),
            "invert": 1,
            "lattice_drift": random.random(),
            "octaves": 2,
            "point_distrib": [m.value for m in grid_dists][random.randint(0, len(grid_dists) - 1)],
            "point_freq": random.randint(7, 10),
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_nth": random.randint(1, 5),
            "warp_octaves": random.randint(1, 3),
            "warp_range": 0 if random.randint(0, 1) else random.random() * .5,
            "with_bloom": 0.25 + random.random() * .5,
            "with_sobel": 1,
            "with_voronoi": 5,
        }
    },

    "web-of-lies": {
        "kwargs": {
            "point_distrib": (["spiral"] + [m.value for m in circular_dists])[random.randint(0, len(circular_dists))],
            "point_freq": 10,
            "voronoi_alpha": 0.25 + random.random() * .5,
            "voronoi_refract": random.randint(1, 3),
            "with_bloom": 0.25 + random.random() * .5,
            "with_shadow": .5,
            "with_voronoi": random.randint(1, 2),
         }
     },

    "wild-kingdom": {
        "kwargs": {
            "freq": 25,
            "lattice_drift": 1,
            "mask": "sparse",
            "post_hue_rotation": random.random(),
            "posterize_levels": 3,
            "rgb": True,
            "ridges": True,
            "with_bloom": 2.0,
            "warp_octaves": 2,
            "warp_range": .05,
            "with_outline": 2,
        }
    },

    "woahdude-voronoi-refract": {
        "kwargs": {
            "freq": 4,
            "hue_range": 2,
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
            "hue_range": random.random() * 3.0,
            "sin": random.randint(5, 15),
            "warp_range": random.randint(3, 5),
            "warp_octaves": 3,
            "with_bloom": .25 + random.random() * .5,
            "with_shadow": random.random(),
        }
    },

    "wooly-bully": {
        "kwargs": {
            "hue_range": random.random() * 1.5,
            "point_corners": True,
            "point_distrib": [m.value for m in circular_dists][random.randint(0, len(circular_dists) - 1)],
            "point_freq": random.randint(2, 3),
            "point_generations": 2,
            "reverb_iterations": random.randint(1, 2),
            "refract_range": random.randint(0, 2),
            "voronoi_func": 3,
            "voronoi_nth": random.randint(1, 3),
            "voronoi_alpha": .5 + random.random() * .5,
            "with_reverb": random.randint(0, 2),
            "with_voronoi": 2,
            "with_worms": 4,
            "worms_alpha": .75 + random.random() * .25,
            "worms_density": 250 + random.random() * 250,
            "worms_duration": 1 + random.random() * 1.5,
            "worms_kink": 5 + random.random() * 2.0,
            "worms_stride": 2.5,
            "worms_stride_deviation": 1.25,
        }
    },

    "wormstep": {
        "kwargs": {
            "corners": True,
            "freq": random.randint(2, 4),
            "lattice_drift": random.randint(0, 1),
            "octaves": random.randint(1, 3),
            "with_bloom": .25 + random.random() * .25,
            "with_worms": 4,
            "worms_alpha": .5 + random.random() * .5,
            "worms_density": 500,
            "worms_kink": 1.0 + random.random() * 4.0,
            "worms_stride": 8.0 + random.random() * 4.0,
        }
    },

}


def load(preset_name, preset_set=None):
    """
    Load a named preset. Specify "random" for a random preset.

    Returns a tuple of (dict, dict, str): `generators.multires` keyword args, `recipes.post_process` keyword args, and preset name.

    See the `artmaker` script for an example of how these values are used.

    :param str preset_name: Name of the preset. If "random" is given, a random preset is returned.
    :param dict|None preset_set: Use a provided preset set. Defaults to `presets.PRESETS`.
    :return: tuple(dict, dict, str)
    """

    if preset_set is None:
        preset_set = PRESETS

    if preset_name == "random":
        preset_name = list(preset_set)[random.randint(0, len(preset_set) - 1)]

        preset = preset_set.get(preset_name)

    else:
        preset = preset_set.get(preset_name, {})

    return preset.get("kwargs", {}), preset.get("post_kwargs", {}), preset_name
