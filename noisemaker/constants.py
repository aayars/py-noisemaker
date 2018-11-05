from enum import Enum

import numpy as np


class ConvKernel(Enum):
    """
    A collection of convolution kernels for image post-processing, based on well-known recipes.

    Pass the desired kernel as an argument to :py:func:`convolve`.

    .. code-block:: python

       image = convolve(ConvKernel.shadow, image)
    """

    invert = [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ]

    emboss = [
        [0, 2, 4],
        [-2, 1, 2],
        [-4, -2, 0]
    ]

    rand = np.random.normal(.5, .5, (5, 5)).tolist()

    edges = [
        [1, 2, 1],
        [2, -12, 2],
        [1, 2, 1]
    ]

    sharpen = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]

    sobel_x = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]

    sobel_y = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]

    deriv_x = [
        [0, 0, 0],
        [0, 1, -1],
        [0, 0, 0]
    ]

    deriv_y = [
        [0, 0, 0],
        [0, 1, 0],
        [0, -1, 0]
    ]

    blur = [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ]


class DistanceFunction(Enum):
    """
    Specify the distance function used in various operations, such as Voronoi cells, derivatives, and sobel operators.
    """

    none = 0

    euclidean = 1

    manhattan = 2

    chebyshev = 3


class InterpolationType(Enum):
    """
    Specify the spline point count for interpolation operations.
    """

    #:
    constant = 0

    #:
    linear = 1

    #:
    cosine = 2

    #:
    bicubic = 3


class PointDistribution(Enum):
    """
    Point cloud distribution, used by Voronoi and DLA
    """

    random = 0

    square = 1

    waffle = 2

    chess = 3

    h_hex = 10

    v_hex = 11

    spiral = 50

    circular = 100

    concentric = 101

    rotating = 102

    @classmethod
    def grid_members(cls):
        return [m for m in cls if cls.is_grid(m)]

    @classmethod
    def circular_members(cls):
        return [m for m in cls if cls.is_circular(m)]

    @classmethod
    def is_grid(cls, member):
        return member.value >= cls.square.value and member.value < cls.spiral.value

    @classmethod
    def is_circular(cls, member):
        return member.value >= cls.circular.value


class ValueDistribution(Enum):
    """
    Specify the value distribution function for basic noise.

    .. code-block:: python

       image = basic(freq, [height, width, channels], distrib=ValueDistribution.uniform)
    """

    normal = 0

    uniform = 1

    exp = 2

    laplace = 3

    lognormal = 4

    ones = 5

    mids = 6


class ValueMask(Enum):
    """
    """

    square = 1
    waffle = 2
    chess = 3

    h_hex = 10
    v_hex = 11
    h_tri = 12
    v_tri = 13

    zero = 20
    one = 21
    two = 22
    three = 23
    four = 24
    five = 25
    six = 26
    seven = 27
    eight = 28
    nine = 29
    a = 30
    b = 31
    c = 32
    d = 33
    e = 34
    f = 35

    tromino_i = 40
    tromino_l = 41
    tromino_o = 42
    tromino_s = 43

    halftone_0 = 50
    halftone_1 = 51
    halftone_2 = 52
    halftone_3 = 53
    halftone_4 = 54
    halftone_5 = 55
    halftone_6 = 56
    halftone_7 = 57
    halftone_8 = 58
    halftone_9 = 59

    lcd_0 = 60
    lcd_1 = 61
    lcd_2 = 62
    lcd_3 = 63
    lcd_4 = 64
    lcd_5 = 65
    lcd_6 = 66
    lcd_7 = 67
    lcd_8 = 68
    lcd_9 = 69  # nice

    fat_lcd_0 = 70
    fat_lcd_1 = 71
    fat_lcd_2 = 72
    fat_lcd_3 = 73
    fat_lcd_4 = 74
    fat_lcd_5 = 75
    fat_lcd_6 = 76
    fat_lcd_7 = 77
    fat_lcd_8 = 78
    fat_lcd_9 = 79
    fat_lcd_a = 80
    fat_lcd_b = 81
    fat_lcd_c = 82
    fat_lcd_d = 83
    fat_lcd_e = 84
    fat_lcd_f = 85
    fat_lcd_g = 86
    fat_lcd_h = 87
    fat_lcd_i = 88
    fat_lcd_j = 89
    fat_lcd_k = 90
    fat_lcd_l = 91
    fat_lcd_m = 92
    fat_lcd_n = 93
    fat_lcd_o = 94
    fat_lcd_p = 95
    fat_lcd_q = 96
    fat_lcd_r = 97
    fat_lcd_s = 98
    fat_lcd_t = 99
    fat_lcd_u = 100
    fat_lcd_v = 101
    fat_lcd_w = 102
    fat_lcd_x = 103
    fat_lcd_y = 104
    fat_lcd_z = 105

    sparse = 1000

    invaders = 1001

    invaders_square = 1002

    matrix = 1003

    letters = 1004

    ideogram = 1005

    iching = 1006

    script = 1007

    white_bear = 1008

    binary = 1009

    tromino = 1010

    numeric = 1011

    hex = 1012

    truetype = 1020

    halftone = 1021

    lcd = 1022

    fat_lcd = 1023

    @classmethod
    def grid_members(cls):
        return [m for m in cls if cls.is_grid(m)]

    @classmethod
    def is_grid(cls, member):
        return member.value < cls.zero.value

    @classmethod
    def procedural_members(cls):
        return [m for m in cls if cls.is_procedural(m)]

    @classmethod
    def is_procedural(cls, member):
        return member.value >= cls.sparse.value


class VoronoiDiagramType(Enum):
    """
    Specify the artistic rendering function used for Voronoi diagrams.
    """

    #: No Voronoi
    none = 0

    #: Normalized neighbor distances
    range = 1

    #: Normalized neighbor distances blended with input Tensor
    color_range = 2

    #: Indexed regions
    regions = 3

    #: Color-mapped regions
    color_regions = 4

    #: Colorized neighbor distances blended with color-mapped regions
    range_regions = 5

    #: Edgeless voronoi. Natural logarithm of reduced distance sums.
    flow = 6

    #: Density-mapped flow diagram
    density = 7

    #: Stitched collage based on indexed regions
    collage = 8

    @classmethod
    def flow_members(cls):
        return [cls.flow, cls.density]

    @classmethod
    def is_flow_member(cls, member):
        return member in cls.flow_members()


class WormBehavior(Enum):
    """
    Specify the type of heading bias for worms to follow.

    .. code-block:: python

       image = worms(image, behavior=WormBehavior.unruly)
    """

    none = 0

    obedient = 1

    crosshatch = 2

    unruly = 3

    chaotic = 4

    random = 5
