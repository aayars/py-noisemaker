from enum import Enum

import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


class Distribution(Enum):
    """
    Specify the type of random distribution for basic noise.

    .. code-block:: python

       image = basic(freq, width, height, channels, dist=Distribution.uniform)
    """

    normal = 0

    uniform = 1

    exponential = 2


def basic(freq, width, height, channels, ridged=False, wavelet=False, spline_order=3, seed=None,
          dist=Distribution.normal, **post_process_args):
    """
    Generate a single layer of scaled noise.

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
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int|Distribution dist: Type of noise distribution. 0=Normal, 1=Uniform, 2=Exponential
    :param int seed: Random seed for reproducible output. Ineffective with exponential.
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(dist, int):
        dist = Distribution(dist)

    initial_shape = [freq, int(freq * width / height), channels]

    if dist == Distribution.normal:
        tensor = tf.random_normal(initial_shape, seed=seed)

    elif dist == Distribution.uniform:
        tensor = tf.random_uniform(initial_shape, seed=seed)

    elif dist == Distribution.exponential:
        tensor = np.random.exponential(size=initial_shape)

    if wavelet:
        tensor = effects.wavelet(tensor)

    tensor = effects.resample(tensor, width, height, spline_order=spline_order)

    if ridged:
        tensor = effects.crease(tensor)

    return effects.post_process(tensor, **post_process_args)


def multires(freq, width, height, channels, octaves, ridged=True, wavelet=True, spline_order=3, seed=None,
             layer_refract_range=0.0, layer_reindex_range=0.0, dist=0, **post_process_args):
    """
    Generate multi-resolution value noise. For each octave: freq increases, amplitude decreases.

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
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int seed: Random seed for reproducible output. Ineffective with exponential.
    :param float layer_refract_range: Per-octave self-distort gradient.
    :param float layer_reindex_range: Per-octave self-reindexing gradient.
    :param int|Distribution dist: Type of noise distribution. 0=Normal, 1=Uniform, 2=Exponential
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    tensor = tf.zeros([height, width, channels])

    for octave in range(1, octaves + 1):
        base_freq = int(freq * .5 * 2**octave)

        if base_freq > width and base_freq > height:
            break

        layer = basic(base_freq, width, height, channels, ridged=ridged, wavelet=wavelet, spline_order=spline_order, seed=seed,
                      refract_range=layer_refract_range, reindex_range=layer_reindex_range, dist=dist)

        tensor += layer / 2**octave

    tensor = effects.normalize(tensor)

    return effects.post_process(tensor, **post_process_args)
