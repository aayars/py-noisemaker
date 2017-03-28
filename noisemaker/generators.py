import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


def gaussian(freq, width, height, channels, ridged=False, wavelet=False, displacement=0.0, spline_order=3):
    """
    Generate scaled noise with a normal distribution.

    :param int freq: Noise frequency per image height
    :param int width: Image output width
    :param int height: Image output height
    :param int channels: Channel count. 1=Gray, 3=RGB, others may not work.
    :param bool ridged: "Crease" in the middle. (1 - unsigned((n-.5)*2))
    :param bool wavelet: Maybe not wavelets this time?
    :param float displacement: Self-displacement gradient. Current implementation is slow.
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :return: Tensor
    """

    tensor = tf.random_normal([freq, int(freq * width / height), channels])

    if wavelet:
        tensor = effects.wavelet(tensor)

    tensor = effects.resample(tensor, width, height, spline_order=spline_order)

    if displacement != 0:
        tensor = effects.displace(tensor, displacement=displacement)

    if ridged:
        tensor = effects.crease(tensor)

    return effects.normalize(tensor)


def multires(freq, width, height, channels, octaves, ridged=True, wavelet=True, displacement=0.0, spline_order=3):
    """
    Generate multi-resolution value noise from a gaussian basis. For each octave: freq increases, amplitude decreases.

    :param int freq: Noise frequency per image height
    :param int width: Image output width
    :param int height: Image output height
    :param int channels: Channel count. 1=Gray, 3=RGB, others may not work.
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8.
    :param bool ridged: "Crease" in the middle. (1 - unsigned((n-.5)*2))
    :param bool wavelet: Maybe not wavelets this time?
    :param float displacement: Self-displacement gradient. Current implementation is slow.
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :return: Tensor
    """

    tensor = tf.zeros([height, width, channels])

    for octave in range(1, octaves + 1):
        base_freq = int(freq * .5 * 2**octave)

        if base_freq * 2 >= width or base_freq * 2 >= height:
            break

        layer = gaussian(base_freq, width, height, channels, ridged=ridged, wavelet=wavelet, spline_order=spline_order)

        tensor = tf.add(tensor, tf.divide(layer, 2**octave))

    if displacement != 0:
        tensor = effects.displace(tensor, displacement=displacement)

    if channels > 2:
        tensor = tf.image.adjust_saturation(tensor, .5)

    return effects.normalize(tensor)
