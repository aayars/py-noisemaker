import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


def gaussian(freq, width, height, channels, ridged=False, wavelet=False, refract=0.0, reindex=0.0,
             clut=None, clut_range=.5, horizontal=False, worms=False, spline_order=3, seed=None):
    """
    Generate scaled noise with a normal distribution.

    .. image:: images/gaussian.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int freq: Heightwise noise frequency
    :param int width: Image output width
    :param int height: Image output height
    :param int channels: Channel count. 1=Gray, 3=RGB, others may not work.
    :param bool ridged: "Crease" at midpoint values: (1 - unsigned((n-.5)*2))
    :param bool wavelet: Maybe not wavelets this time?
    :param float refract: Self-distortion gradient.
    :param float reindex: Self-reindexing gradient.
    :param str clut: PNG or JPG color lookup table filename.
    :param float horizontal: Preserve clut Y axis.
    :param float clut_range: Gather range for clut.
    :param bool worms: Do worms.
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int seed: Random seed for reproducible output.
    :return: Tensor
    """

    tensor = tf.random_normal([freq, int(freq * width / height), channels], seed=seed)

    if wavelet:
        tensor = effects.wavelet(tensor)

    tensor = effects.resample(tensor, width, height, spline_order=spline_order)

    if ridged:
        tensor = effects.crease(tensor)

    return effects.post_process(tensor, refract, reindex, clut, horizontal, clut_range, worms)


def multires(freq, width, height, channels, octaves, ridged=True, wavelet=True,
             refract=0.0, layer_refract=0.0, reindex=0.0, layer_reindex=0.0, clut=None,
             clut_range=.5, horizontal=False, worms=False, spline_order=3, seed=None):
    """
    Generate multi-resolution value noise from a gaussian basis. For each octave: freq increases, amplitude decreases.

    .. image:: images/multires.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int freq: Heightwise bottom layer frequency
    :param int width: Image output width
    :param int height: Image output height
    :param int channels: Channel count. 1=Gray, 3=RGB, others may not work.
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8.
    :param bool ridged: "Crease" at midpoint values: (1 - unsigned((n-.5)*2))
    :param bool wavelet: Maybe not wavelets this time?
    :param float refract: Self-distortion gradient.
    :param float layer_refract: Per-octave self-distort gradient.
    :param float reindex: Self-reindexing gradient.
    :param float layer_reindex: Per-octave self-reindexing gradient.
    :param str clut: PNG or JPG color lookup table filename.
    :param float horizontal: Preserve clut Y axis.
    :param float clut_range: Gather range for clut.
    :param bool worms: Do worms.
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int seed: Random seed for reproducible output.
    :return: Tensor
    """

    tensor = tf.zeros([height, width, channels])

    for octave in range(1, octaves + 1):
        base_freq = int(freq * .5 * 2**octave)

        if base_freq > width and base_freq > height:
            break

        layer = gaussian(base_freq, width, height, channels, ridged=ridged, wavelet=wavelet, spline_order=spline_order, seed=seed,
                         refract=layer_refract, reindex=layer_reindex)

        tensor = tf.add(tensor, tf.divide(layer, 2**octave))

    tensor = effects.normalize(tensor)

    return effects.post_process(tensor, refract, reindex, clut, horizontal, clut_range, worms)
