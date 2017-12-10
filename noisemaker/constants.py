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

    sparse = 100

    invaders = 101

    invaders_square = 102

    matrix = 103

    letters = 104

    ideogram = 105

    iching = 106

    script = 107

    @classmethod
    def grid_members(cls):
        return [m for m in cls if cls.is_grid(m)]

    @classmethod
    def is_grid(cls, member):
        return member.value < cls.sparse.value

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
