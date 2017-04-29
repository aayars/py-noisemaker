from enum import Enum

import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


class Distribution(Enum):
    """
    Specify the random distribution function for basic noise.

    See also: https://docs.scipy.org/doc/numpy/reference/routines.random.html

    .. code-block:: python

       image = basic(freq, [height, width, channels], distrib=Distribution.uniform)
    """

    normal = 0

    uniform = 1

    exponential = 2

    laplace = 3

    lognormal = 4


def basic(freq, shape, ridged=False, wavelet=False, spline_order=3, seed=None,
          distrib=Distribution.normal, **post_process_args):
    """
    Generate a single layer of scaled noise.

    .. image:: images/gaussian.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Base noise frequency. Int, or list of ints for each spatial dimension.
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels].
    :param bool ridged: "Crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int|Distribution distrib: Type of noise distribution. See :class:`Distribution` enum.
    :param int seed: Random seed for reproducible output. Ineffective with exponential.
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(freq, int):
        freq = _freq_for_shape(freq, shape)

    initial_shape = [*freq, shape[-1]]

    if isinstance(distrib, int):
        distrib = Distribution(distrib)

    if distrib == Distribution.normal:
        tensor = tf.random_normal(initial_shape, seed=seed)

    elif distrib == Distribution.uniform:
        tensor = tf.random_uniform(initial_shape, seed=seed)

    elif distrib == Distribution.exponential:
        tensor = tf.stack(np.random.exponential(size=initial_shape))

    elif distrib == Distribution.laplace:
        tensor = tf.stack(np.random.laplace(size=initial_shape))

    elif distrib == Distribution.lognormal:
        tensor = tf.stack(np.random.lognormal(size=initial_shape))

    if wavelet:
        tensor = effects.wavelet(tensor, initial_shape)

    tensor = effects.resample(tensor, shape[0:-1], spline_order=spline_order)

    tensor = effects.post_process(tensor, shape, **post_process_args)

    tensor = effects.normalize(tensor)

    if ridged:
        tensor = effects.crease(tensor)

    return tensor


def multires(freq, shape, octaves=4, ridged=True, wavelet=False, spline_order=3, seed=None,
             layer_refract_range=0.0, layer_reindex_range=0.0, distrib=Distribution.normal, deriv=False,
             **post_process_args):
    """
    Generate multi-resolution value noise. For each octave: freq increases, amplitude decreases.

    .. image:: images/multires.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Bottom layer frequency. Int, or list of ints for each spatial dimension.
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels].
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8.
    :param bool ridged: "Crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 3=Bicubic, others may not work.
    :param int seed: Random seed for reproducible output. Ineffective with exponential.
    :param float layer_refract_range: Per-octave self-distort gradient.
    :param float layer_reindex_range: Per-octave self-reindexing gradient.
    :param int|Distribution distrib: Type of noise distribution. See :class:`Distribution` enum.
    :param bool deriv: Derivative noise.
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    tensor = tf.zeros(shape)

    if isinstance(freq, int):
        freq = _freq_for_shape(freq, shape)

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = basic(base_freq, shape, ridged=ridged, wavelet=wavelet, spline_order=spline_order, seed=seed,
                      refract_range=layer_refract_range / multiplier, reindex_range=layer_reindex_range / multiplier,
                      distrib=distrib, deriv=deriv)

        tensor += layer / multiplier

    tensor = effects.normalize(tensor)

    return effects.post_process(tensor, shape, **post_process_args)


def _freq_for_shape(freq, shape):
    """
    Given a base frequency as int, generate noise frequencies for each spatial dimension.

    :param int freq: Base frequency
    :param list[int] shape: List of spatial dimensions, e.g. [height, width]
    """

    return [int(freq * shape[i] / shape[0]) for i in range(len(shape) - 1)]