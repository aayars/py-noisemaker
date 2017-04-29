from enum import Enum

import math
import random

import numpy as np
import tensorflow as tf


def post_process(tensor, refract_range=0.0, reindex_range=0.0, clut=None, clut_horizontal=False, clut_range=0.5,
                 with_worms=False, worm_behavior=None, worm_density=4.0, worm_duration=4.0, worm_stride=1.0, worm_stride_deviation=.05,
                 worm_bg=.5, with_sobel=False, with_normal_map=False, deriv=False, with_glitch=False):
    """
    Apply post-processing filters.

    :param Tensor tensor:
    :param float refract_range: Self-distortion gradient.
    :param float reindex_range: Self-reindexing gradient.
    :param str clut: PNG or JPG color lookup table filename.
    :param float clut_horizontal: Preserve clut Y axis.
    :param float clut_range: Gather range for clut.
    :param bool with_worms: Do worms.
    :param WormBehavior worm_behavior:
    :param float worm_density: Worm density multiplier (larger == slower)
    :param float worm_duration: Iteration multiplier (larger == slower)
    :param float worm_stride: Mean travel distance per iteration
    :param float worm_stride_deviation: Per-worm travel distance deviation
    :param float worm_bg: Background color brightness for worms
    :param bool with_sobel: Sobel operator
    :param bool with_normal_map: Create a tangent-space normal map
    :param bool deriv: Derivative operator
    :param bool with_glitch: Bit shit
    :return: Tensor
    """

    if refract_range != 0:
        tensor = refract(tensor, displacement=refract_range)

    if reindex_range != 0:
        tensor = reindex(tensor, displacement=reindex_range)

    if clut:
        tensor = color_map(tensor, clut, horizontal=clut_horizontal, displacement=clut_range)

    else:
        tensor = normalize(tensor)

    if with_worms:
        tensor = worms(tensor, behavior=worm_behavior, density=worm_density, duration=worm_duration,
                       stride=worm_stride, stride_deviation=worm_stride_deviation, bg=worm_bg)

    if deriv:
        tensor = derivative(tensor)

    if with_sobel:
        tensor = sobel(tensor)

    if with_glitch:
        tensor = glitch(tensor)

    if with_normal_map:
        tensor = normal_map(tensor)

    return tensor


class WormBehavior(Enum):
    """
    Specify the type of heading bias for worms to follow.

    .. code-block:: python

       image = worms(image, behavior=WormBehavior.unruly)
    """

    obedient = 0

    crosshatch = 1

    unruly = 2

    chaotic = 3


class ConvKernel(Enum):
    """
    A collection of convolution kernels for image post-processing, based on well-known recipes.

    Pass the desired kernel as an argument to :py:func:`convolve`.

    .. code-block:: python

       image = convolve(ConvKernel.shadow, image)
    """

    emboss = [
        [   0,   2,   4   ],
        [  -2,   1,   2   ],
        [  -4,  -2,   0   ]
    ]

    rand = np.random.normal(.5, .5, (5, 5)).tolist()

    shadow = [
        # yeah, one of the really big fuckers
        # [  0,   1,   1,   1,   1,   1,   1  ],
        # [ -1,   0,   2,   2,   1,   1,   1  ],
        # [ -1,  -2,   0,   4,   2,   1,   1  ],
        # [ -1,  -2,  -4,   12,   4,   2,   1  ],
        # [ -1,  -1,  -2,  -4,   0,   2,   1  ],
        # [ -1,  -1,  -1,  -2,  -2,   0,   1  ],
        # [ -1,  -1,  -1,  -1,  -1,  -1,   0  ]

        # [  0,  1,  1,  1, 0 ],
        # [ -1, -2,  4,  2, 1 ],
        # [ -1, -4,  2,  4, 1 ],
        # [ -1, -2, -4,  2, 1 ],
        # [  0, -1, -1, -1, 0 ]

        [  0,  1,  1,  1, 0 ],
        [ -1, -2,  4,  2, 1 ],
        [ -1, -4,  2,  4, 1 ],
        [ -1, -2, -4,  2, 1 ],
        [  0, -1, -1, -1, 0 ]
    ]

    edges = [
        [   1,   2,  1   ],
        [   2, -12,  2   ],
        [   1,   2,  1   ]
    ]

    sharpen = [
        [   0, -1,  0 ],
        [  -1,  5, -1 ],
        [   0, -1,  0 ]
    ]

    unsharp_mask = [
        [ 1,  4,     6,   4, 1 ],
        [ 4,  16,   24,  16, 4 ],
        [ 6,  24, -476,  24, 6 ],
        [ 4,  16,   24,  16, 4 ],
        [ 1,  4,     6,   4, 1 ]
    ]

    invert = [
        [ 0,  0,  0 ],
        [ 0, -1,  0 ],
        [ 0,  0,  0 ]
    ]

    sobel_x = [
        [ 1, 0, -1 ],
        [ 2, 0, -2 ],
        [ 1, 0, -1 ]
    ]

    sobel_y = [
        [  1,  2,  1 ],
        [  0,  0,  0 ],
        [ -1, -2, -1 ]
    ]


def _conform_kernel_to_tensor(kernel, tensor):
    """ Re-shape a convolution kernel to match the given tensor's color dimensions. """

    l = len(kernel)

    channels = tf.shape(tensor).eval()[-1]

    temp = np.repeat(kernel, channels)

    temp = tf.reshape(temp, (l, l, channels, 1))

    temp = tf.image.convert_image_dtype(temp, tf.float32)

    return temp


def convolve(kernel, tensor):
    """
    Apply a convolution kernel to an image tensor.

    :param ConvKernel kernel: See ConvKernel enum
    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    shape = tf.shape(tensor)
    height, width, channels = shape.eval()

    kernel_values = _conform_kernel_to_tensor(kernel.value, tensor)

    # Give the conv kernel some room to play on the edges
    half_height = tf.cast(shape[0] / 2, tf.int32)
    half_width = tf.cast(shape[1] / 2, tf.int32)

    tensor = tf.tile(tensor, [3, 3, 1])  # Tile 3x3
    tensor = tensor[half_height:shape[0] * 2 + half_height, half_width:shape[1] * 2 + half_width]  # Center Crop 2x2
    tensor = tf.nn.depthwise_conv2d([tensor], kernel_values, [1,1,1,1], "VALID")[0]
    tensor = tensor[half_height:shape[0] + half_height, half_width:shape[1] + half_width]  # Center Crop 1x1
    tensor = normalize(tensor)

    if kernel == ConvKernel.edges:
        tensor = tf.abs(tensor - .5) * 2

    return tensor


def normalize(tensor):
    """
    Squeeze the given Tensor into a range between 0 and 1.

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor))


def resample(tensor, shape, spline_order=3):
    """
    Resize the given image Tensor to the given dimensions, wrapping around edges.

    :param Tensor tensor: An image tensor.
    :param list[int] shape: The desired shape, specify spatial dimensions only. e.g. [height, width]
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :return: Tensor
    """

    if spline_order == 0:
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif spline_order == 1:
        resize_method = tf.image.ResizeMethod.BILINEAR
    else:
        resize_method = tf.image.ResizeMethod.BICUBIC

    input_shape = tf.shape(tensor)
    half_input_height = tf.cast(input_shape[0] / 2, tf.int32)
    half_input_width = tf.cast(input_shape[1] / 2, tf.int32)

    half_height = tf.cast(shape[0] / 2, tf.int32)
    half_width = tf.cast(shape[1] / 2, tf.int32)

    tensor = tf.tile(tensor, [3 for d in range(len(shape))] + [1])  # Tile 3x3
    tensor = tensor[half_input_height:input_shape[0] * 2 + half_input_height, half_input_width:input_shape[1] * 2 + half_input_width]  # Center Crop 2x2
    tensor = tf.image.resize_images(tensor, [d * 2 for d in shape], method=resize_method)  # Upsample
    tensor = tensor[half_height:shape[0] + half_height, half_width:shape[1] + half_width]  # Center Crop 1x1
    tensor = tf.image.convert_image_dtype(tensor, tf.float32, saturate=True)

    return tensor


def crease(tensor):
    """
    Create a "crease" (ridge) at midpoint values. 1 - abs(n * 2 - 1)

    .. image:: images/crease.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    return 1 - tf.abs(tensor * 2 - 1)


def derivative(tensor):
    """
    Extract a derivative from the given noise.

    .. image:: images/derived.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :return: Tensor
    """

    y, x = np.gradient(tensor.eval(), axis=(0, 1))  # Do this in TF with conv2D?

    return normalize(tf.sqrt(y*y + x*x))


def reindex(tensor, displacement=.5):
    """
    Re-color the given tensor, by sampling along one axis at a specified frequency.

    .. image:: images/reindex.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param float displacement:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    reference = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    mod = min(height, width)
    offset = tf.cast(tf.mod(tf.add(tf.multiply(reference, displacement * mod), reference), mod), tf.int32)

    temp = tf.reshape(tensor.eval()[offset.eval(), 0], shape)  # XXX Do this with TF

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def refract(tensor, displacement=.5):
    """
    Apply self-displacement along X and Y axes, based on each pixel value.

    .. image:: images/refract.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :param float displacement:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    index = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    # Create two channels for X and Y
    index = tf.reshape(index, [width * height])
    index = np.repeat((index.eval() - .5) * 2 * displacement * min(width, height), 2)
    index = tf.reshape(index, [height, width, 2]).eval()

    index = _offset_index(index)

    row_identity = _row_index(tensor)
    column_identity = _column_index(tensor)

    index[:,:,0] = (index[:,:,0] + column_identity) % height   # XXX Assemble a new Tensor here instead of using nump
    index[:,:,1] = (index[:,:,1] + row_identity) % width

    index = tf.cast(index, tf.int32)

    return tf.gather_nd(tensor, index)


def color_map(tensor, clut, horizontal=False, displacement=.5):
    """
    Apply a color map to an image tensor.

    The color map can be a photo or whatever else.

    .. image:: images/color_map.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param Tensor|str clut: An image tensor or filename (png/jpg only) to use as a color palette
    :param bool horizontal: Scan horizontally
    :param float displacement: Gather distance for clut
    """

    if isinstance(clut, str):
        with open(clut, "rb") as fh:
            if clut.endswith(".png"):
                clut = tf.image.decode_png(fh.read(), channels=3)

            elif clut.endswith(".jpg"):
                clut = tf.image.decode_jpeg(fh.read(), channels=3)

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    # TODO: Reduce tensor to single channel more reliably (use a reduce function?)
    orig_index = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    index = tf.reshape(orig_index, [width * height])
    index *= displacement
    index = np.repeat(index.eval(), 2)
    index = tf.reshape(index, [height, width, 2]).eval()

    row_identity = _row_index(tensor)
    index[:,:,1] = (index[:,:,1] * (width - 1) + row_identity) % width

    column_identity = _column_index(tensor)

    if horizontal:
        index[:,:,0] = column_identity

    else:
        index[:,:,0] = (index[:,:,0] * (height - 1) + column_identity) % height
        index = _offset_index(index)

    index = tf.cast(index, tf.int32)

    clut = resample(clut, [height, width], 3)

    clut = tf.image.convert_image_dtype(clut, tf.float32, saturate=True)

    output = tf.gather_nd(clut, index)
    output = tf.image.convert_image_dtype(output, tf.float32, saturate=True)

    return output


def worms(tensor, behavior=0, density=4.0, duration=4.0, stride=1.0, stride_deviation=.05, bg=.5):
    """
    Make a furry patch of worms which follow field flow rules.

    .. image:: images/worms.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor:
    :param int|WormBehavior behavior:
    :param float density: Worm density multiplier (larger == slower)
    :param float duration: Iteration multiplier (larger == slower)
    :param float stride: Mean travel distance per iteration
    :param float stride_deviation: Per-worm travel distance deviation
    :param float bg: Background color intensity.
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    reference = tensor.eval()

    count = int(max(width, height) * density)
    worms = np.random.uniform(size=[count, 4])  # Worm: [ y, x, rotation bias, stride ]
    worms[:, 0] *= height
    worms[:, 1] *= width
    worms[:, 3] = np.random.normal(loc=stride, scale=stride_deviation, size=[count])

    if isinstance(behavior, int):
        behavior = WormBehavior(behavior)

    if behavior == WormBehavior.obedient:
        worms[:, 2] = 0

    elif behavior == WormBehavior.crosshatch:
        worms[:, 2] = np.mod(np.floor(worms[:, 2] * 100), 2) * 90

    elif behavior == WormBehavior.chaotic:
        worms[:, 2] *= 360.0

    colors = tf.zeros((count, channels)).eval()
    for i, worm in enumerate(worms):
        colors[i] = reference[int(worm[0]) % height, int(worm[1]) % width]

    index = tf.image.rgb_to_grayscale(reference) if channels > 2 else reference
    index = tf.reshape(index, (height, width))
    index = normalize(index) * 360.0 * math.radians(1)
    index = index.eval()

    iterations = int(math.sqrt(min(width, height)) * duration)

    out = reference * bg

    # Make worms!
    for i in range(iterations):
        coords = worms.astype(int)

        coords[:, 0] = np.mod(coords[:, 0], height)
        coords[:, 1] = np.mod(coords[:, 1], width)

        value = 1 + (1 - abs(1 - i / (iterations - 1) * 2))  # Makes linear gradient [ 1 .. 2 .. 1 ]

        # Stretch goal: Get this out of a loop and do straight up math
        for j, coord in enumerate(coords):
            out[coord[0], coord[1]] = value * colors[j]

        worms[:, 0] += np.cos(index[coords[:, 0], coords[:, 1]] + worms[:, 2]) * worms[:, 3]
        worms[:, 1] += np.sin(index[coords[:, 0], coords[:, 1]] + worms[:, 2]) * worms[:, 3]

    out = tf.image.convert_image_dtype(out, tf.float32, saturate=True)

    return out


def wavelet(tensor):
    """
    Convert regular noise into 2-D wavelet noise.

    Completely useless. Maybe useful if Noisemaker supports higher dimensions later.

    .. image:: images/wavelet.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param Tensor tensor: An image tensor.
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    return normalize(tensor - resample(resample(tensor, [int(height * .5), int(width * .5)]), [height, width]))


def sobel(tensor):
    """
    Apply a sobel operator.

    :param Tensor tensor:
    :return: Tensor
    """

    x = convolve(ConvKernel.sobel_x, tensor)
    y = convolve(ConvKernel.sobel_y, tensor)

    return tf.abs(normalize(tf.sqrt(x * x + y * y)) * 2 - 1)


def normal_map(tensor):
    """
    Generate a tangent-space normal map.

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    reference = tf.image.rgb_to_grayscale(tensor) if channels > 2 else tensor

    x = normalize(1 - convolve(ConvKernel.sobel_x, reference))
    y = normalize(convolve(ConvKernel.sobel_y, reference))
    z = 1 - tf.abs(normalize(tf.sqrt(x * x + y * y)) * 2 - 1) * .5 + .5

    output = np.zeros([height, width, 3])
    output[:,:,0] = x.eval()[:,:,0]
    output[:,:,1] = y.eval()[:,:,0]
    output[:,:,2] = z.eval()[:,:,0]

    return output


def _xy_index(tensor):
    """
    """

    shape = tf.shape(tensor).eval()

    index = np.zeros([*shape[0:-1], 2])

    index[:,:,0] = _column_index(tensor)
    index[:,:,1] = _row_index(tensor)

    return tf.cast(index, tf.int32)


def _row_index(tensor):
    """
    Generate an X index for the given tensor.

    [
      [ 0, 1, 2, ... width-1 ],
      [ 0, 1, 2, ... width-1 ],
      ... (x height)
    ]

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    row_identity = tf.cumsum(tf.ones(width, dtype=tf.int32), exclusive=True)
    row_identity = tf.reshape(tf.tile(row_identity, [height]), [height, width]).eval()

    return row_identity


def _column_index(tensor):
    """
    Generate a Y index for the given tensor.

    [
      [ 0, 0, 0, ... ],
      [ 1, 1, 1, ... ],
      [ n, n, n, ... ],
      ...
      [ height-1, height-1, height-1, ... ]
    ]

    :param Tensor tensor:
    :return: Tensor
    """

    shape = tf.shape(tensor).eval()
    height, width, channels = shape

    column_identity = tf.ones(width, dtype=tf.int32)
    column_identity = tf.tile(column_identity, [height])
    column_identity = tf.reshape(column_identity, [height, width])
    column_identity = tf.cumsum(column_identity, exclusive=True).eval()

    return column_identity


def blend(a, b, g):
    """
    """

    a *= 1 - g
    b *= g

    return a + b


def _offset_index(tensor):
    """
    Offset X and Y displacement channels from each other, to help with diagonal banding.

    :param Tensor tensor: Tensor of shape (height, width, 2)
    :return: Tensor
    """

    tensor[:,:,0] = offset_y(tensor[:,:,0])
    tensor[:,:,1] = offset_x(tensor[:,:,1])

    return tensor


def offset_x(tensor):
    """
    """

    shape = tf.shape(tensor).eval()
    width = shape[1]

    # tensor = np.rot90(tensor)
    return np.roll(tensor, int(random.random() * width * .5 + width * .5), axis=1)

    # return np.rot90(tensor, 3)


def offset_y(tensor):
    """
    """

    shape = tf.shape(tensor).eval()
    height = shape[0]

    return np.roll(tensor, int(random.random() * height * .5 + height * .5), axis=0)
