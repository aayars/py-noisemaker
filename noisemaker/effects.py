from enum import Enum

import numpy as np
import tensorflow as tf


class ConvKernel(Enum):
    identity = [
        [   0,  0,  0   ],
        [   0,  1,  0   ],
        [   0,  0,  0   ],
    ]

    shadow = [
        [   0,   2,   4   ],
        [  -2,   1,   2   ],
        [  -4,  -2,   0   ],
    ]

    edges = [
        [   1,   2,  1   ],
        [   2, -12,  2   ],
        [   1,   2,  1   ],
    ]

    sharpen = [
        [   0, -1,  0 ],
        [  -1,  5, -1 ],
        [   0, -1,  0 ],
    ]

    unsharp_mask = [
        [ 1,  4,     6,   4, 1 ],
        [ 4,  16,   24,  16, 4 ],
        [ 6,  24, -476,  24, 6 ],
        [ 4,  16,   24,  16, 4 ],
        [ 1,  4,     6,   4, 1 ],
    ]


def normalize(tensor):
    """
    Squeeze the given Tensor into a range between 0 and 1.
    """

    return tf.divide(
        tf.subtract(tensor, tf.reduce_min(tensor)),
        tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor))
    )


def resample(tensor, width, height):
    """
    Resize the given image Tensor to the given dimensions.
    """

    temp = tf.image.resize_images(tensor, [height, width], align_corners=True, method=tf.image.ResizeMethod.BICUBIC)

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def crease(tensor):
    """
    """

    temp = tf.subtract(tensor, .5)
    temp = tf.multiply(temp, 2)
    temp = tf.maximum(temp, temp*-1)

    temp = tf.subtract(tf.ones(tf.shape(temp)), temp)

    return temp


def conform_kernel_to_tensor(kernel, tensor):
    """
    """

    l = len(kernel)

    channels = tf.shape(tensor).eval()[2]

    temp = np.repeat(kernel, channels)

    temp = tf.reshape(temp, (l, l, channels, 1))

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def convolve(kernel, tensor):
    """
    """

    return tf.nn.depthwise_conv2d([tensor], conform_kernel_to_tensor(kernel, tensor), [1,1,1,1], "VALID")[0]


def displace(tensor, displacement=1.0):
    """
    """

    shape = tf.shape(tensor).eval()

    width, height, channels = shape

    reference = tf.image.rgb_to_grayscale(tensor) if channels > 1 else tensor

    reference = reference.eval()
    tensor = tensor.eval()

    temp = np.zeros(shape)

    for x in range(width):
        for y in range(height):
            value = reference[y][x] * displacement

            x_offset = (x + int(value * width)) % width
            y_offset = (y + int(value * height)) % height

            temp[y][x] = tensor[x_offset][y_offset]

    temp = tf.image.convert_image_dtype(temp, tf.float32, saturate=True)

    return temp


def wavelet(tensor):
    """
    """

    shape = tf.shape(tensor).eval()

    height, width, channels = shape

    return tensor - resample(resample(tensor, int(width * .5), int(height * .5)), width, height)
