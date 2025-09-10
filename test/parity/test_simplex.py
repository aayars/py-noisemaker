import math

from noisemaker import simplex as simplex_module

coords = [i / 7 for i in range(4)]
EXPECTED = {
    1: {
        "perm": [68, 151, 59, 159, 166, 58, 162, 75, 242, 1],
        "perm_grad": [60, 21, 33, 45, 66, 30, 54, 9, 6, 3],
        "first8": [
            0.5,
            0.4575223852942156,
            0.42505810577248015,
            0.4053217856165236,
            0.5429448393766084,
            0.5014467765690047,
            0.4676721563277974,
            0.44654853177583503,
        ],
        "random": 0.5625817649168068,
        "checksum": 25.5527973325285,
    },
    2: {
        "perm": [176, 1, 234, 36, 27, 83, 21, 156, 138, 147],
        "perm_grad": [24, 3, 54, 36, 9, 33, 63, 36, 54, 9],
        "first8": [
            0.5,
            0.5417448253228163,
            0.5659347277375898,
            0.5606383039015482,
            0.45870195865908486,
            0.49958335799743625,
            0.5208144085554277,
            0.508921250577127,
        ],
        "random": 0.4374182350831932,
        "checksum": 38.60484882243753,
    },
    3: {
        "perm": [214, 180, 149, 105, 98, 243, 249, 251, 208, 11],
        "perm_grad": [66, 36, 15, 27, 6, 9, 27, 33, 48, 33],
        "first8": [
            0.5,
            0.45735516358437,
            0.42345124128828066,
            0.4007485040318812,
            0.4575630269720605,
            0.4179466527127099,
            0.3879962718776424,
            0.36840249818386106,
        ],
        "random": 0.36106808044234445,
        "checksum": 32.28227356708986,
    },
}


def test_simplex_parity():
    for seed, exp in EXPECTED.items():
        os, data = simplex_module._from_seed(seed)
        assert list(os._perm[:10]) == exp["perm"]
        assert list(os._perm_grad_index_3D[:10]) == exp["perm_grad"]
        grid = []
        for x in coords:
            for y in coords:
                for z in coords:
                    grid.append((os.noise3d(x, y, z) + 1) / 2)
        for actual, expected in zip(grid[:8], exp["first8"]):
            assert abs(actual - expected) < 1e-6
        assert abs(sum(grid) - exp["checksum"]) < 1e-6
        r = simplex_module.random(0.25, seed, 1)
        assert abs(r - exp["random"]) < 1e-6
        dt = 1e-3
        d0 = (simplex_module.random(dt, seed, 1) - simplex_module.random(0.0, seed, 1)) / dt
        d1 = (simplex_module.random(1 + dt, seed, 1) - simplex_module.random(1.0, seed, 1)) / dt
        assert abs(d0 - d1) < 1e-6
