import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


def gaussian(freq, width, height, channels, ridged=False, wavelet=False, displacement=0.0):
    """
    """

    tensor = tf.random_normal([freq, int(freq * width / height), channels])

    if wavelet:
        tensor = effects.wavelet(tensor)

    tensor = effects.resample(tensor, width, height)

    if displacement != 0:
        tensor = effects.displace(tensor, displacement=displacement)

    if ridged:
        tensor = effects.crease(tensor)

    return effects.normalize(tensor)


def multires(freq, width, height, channels, octaves, ridged=True, wavelet=True, displacement=0.0, layer_displacement=0.0):
    """
    """

    tensor = tf.zeros([height, width, channels])

    for octave in range(1, octaves + 1):
        base_freq = int(freq * .5 * 2**octave)

        if base_freq * 2 >= width or base_freq * 2 >= height:
            break

        layer = gaussian(base_freq, width, height, channels, ridged=ridged, wavelet=wavelet, displacement=layer_displacement)

        tensor = tf.add(tensor, tf.divide(layer, 2**octave))

    if displacement != 0:
        tensor = effects.displace(tensor, displacement=displacement)

    if channels > 2:
        tensor = tf.image.adjust_saturation(tensor, .5)

    return effects.normalize(tensor)
