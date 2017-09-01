from enum import Enum

import numpy as np
import tensorflow as tf

import noisemaker.effects as effects


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

    chess = 5


def basic(freq, shape, ridges=False, sin=0.0, wavelet=False, spline_order=3, seed=None,
          distrib=ValueDistribution.normal, lattice_drift=0.0,
          hsv=True, hsv_range=.125, hsv_rotation=None, hsv_saturation=1.0,
          **post_process_args):
    """
    Generate a single layer of scaled noise.

    .. image:: images/gaussian.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Base noise frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param bool ridges: "Crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int|str|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param float lattice_drift: Push away from underlying lattice
    :param int seed: Random seed for reproducible output. Ineffective with exp
    :param bool hsv: Set to False for RGB noise
    :param float hsv_range: HSV hue range
    :param float|None hsv_rotation: HSV hue bias
    :param float hsv_saturation: HSV saturation
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    initial_shape = [*freq, shape[-1]]

    if isinstance(distrib, int):
        distrib = ValueDistribution(distrib)

    elif isinstance(distrib, str):
        distrib = ValueDistribution[distrib]

    if distrib == ValueDistribution.normal:
        tensor = tf.random_normal(initial_shape, seed=seed)

    elif distrib == ValueDistribution.uniform:
        tensor = tf.random_uniform(initial_shape, seed=seed)

    elif distrib == ValueDistribution.exp:
        tensor = tf.cast(tf.stack(np.random.exponential(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.laplace:
        tensor = tf.cast(tf.stack(np.random.laplace(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.lognormal:
        tensor = tf.cast(tf.stack(np.random.lognormal(size=initial_shape)), tf.float32)

    elif distrib == ValueDistribution.chess:
        channel_shape = [*freq, 1]

        tensor = tf.cast(tf.reshape(effects.row_index(channel_shape) + effects.column_index(channel_shape), channel_shape) % 2, tf.float32) * tf.ones(initial_shape)

    if wavelet:
        tensor = effects.wavelet(tensor, initial_shape)

    tensor = effects.resample(tensor, shape, spline_order=spline_order)

    if lattice_drift:
        displacement = lattice_drift / min(freq[0], freq[1])

        tensor = effects.refract(tensor, shape, displacement=displacement, warp_freq=freq, spline_order=spline_order)

    tensor = effects.post_process(tensor, shape, freq, spline_order=spline_order, **post_process_args)

    if shape[-1] == 3 and hsv:
        if hsv_rotation is None:
            hsv_rotation = tf.random_normal([])

        hue = (tensor[:,:,0] * hsv_range + hsv_rotation) % 1.0

        saturation = effects.normalize(tensor[:,:,1]) * hsv_saturation

        value = effects.crease(tensor[:,:,2]) if ridges else tensor[:,:,2]

        if sin:
            value = effects.normalize(tf.sin(sin * value))

        tensor = tf.image.hsv_to_rgb([tf.stack([hue, saturation, value], 2)])[0]

    elif ridges:
        tensor = effects.crease(tensor)

    if sin and not hsv:
        tensor = tf.sin(sin * tensor)

    return tensor


def multires(freq, shape, octaves=4, ridges=True, sin=0.0, wavelet=False, spline_order=3, seed=None,
             reflect_range=0.0, refract_range=0.0, reindex_range=0.0, distrib=ValueDistribution.normal,
             deriv=False, deriv_func=0, deriv_alpha=1.0, lattice_drift=0.0,
             post_reflect_range=0.0, post_refract_range=0.0, post_deriv=False,
             hsv=True, hsv_range=.125, hsv_rotation=None, hsv_saturation=1.0,
             **post_process_args):
    """
    Generate multi-resolution value noise. For each octave: freq increases, amplitude decreases.

    .. image:: images/multires.jpg
       :width: 1024
       :height: 256
       :alt: Noisemaker example output (CC0)

    :param int|list[int] freq: Bottom layer frequency. Int, or list of ints for each spatial dimension
    :param list[int]: Shape of noise. For 2D noise, this is [height, width, channels]
    :param int octaves: Octave count. Number of multi-res layers. Typically 1-8
    :param bool ridges: "Crease" at midpoint values: (1 - abs(n * 2 - 1))
    :param float sin: Apply sin function to noise basis
    :param bool wavelet: Maybe not wavelets this time?
    :param int spline_order: Spline point count. 0=Constant, 1=Linear, 2=Cosine, 3=Bicubic
    :param int seed: Random seed for reproducible output. Ineffective with exponential
    :param float reflect_range: Per-octave derivative-based distort gradient
    :param float refract_range: Per-octave self-distort gradient
    :param float reindex_range: Per-octave self-reindexing gradient
    :param int|ValueDistribution distrib: Type of noise distribution. See :class:`ValueDistribution` enum
    :param bool deriv: Extract derivatives from noise
    :param DistanceFunction|int deriv_func: Derivative distance function
    :param float deriv_alpha: Derivative alpha blending amount
    :param float lattice_drift: Push away from underlying lattice
    :param float post_reflect_range: Reduced derivative-based distort gradient
    :param float post_refract_range: Reduced self-distort gradient
    :param bool post_deriv: Reduced derivatives
    :param bool hsv: Set to False for RGB noise
    :param float hsv_range: HSV hue range
    :param float|None hsv_rotation: HSV hue bias
    :param float hsv_saturation: HSV saturation
    :return: Tensor

    Additional keyword args will be sent to :py:func:`noisemaker.effects.post_process`
    """

    tensor = tf.zeros(shape)

    if isinstance(freq, int):
        freq = effects.freq_for_shape(freq, shape)

    for octave in range(1, octaves + 1):
        multiplier = 2 ** octave

        base_freq = [int(f * .5 * multiplier) for f in freq]

        if all(base_freq[i] > shape[i] for i in range(len(base_freq))):
            break

        layer = basic(base_freq, shape, ridges=ridges, sin=sin, wavelet=wavelet, spline_order=spline_order, seed=seed,
                      reflect_range=reflect_range / multiplier, refract_range=refract_range / multiplier, reindex_range=reindex_range / multiplier,
                      distrib=distrib, deriv=deriv, deriv_func=deriv_func, deriv_alpha=deriv_alpha, lattice_drift=lattice_drift,
                      hsv=hsv, hsv_range=hsv_range, hsv_rotation=hsv_rotation, hsv_saturation=hsv_saturation,
                      )

        tensor += layer / multiplier

    tensor = effects.post_process(tensor, shape, freq, ridges_hint=ridges and not hsv, spline_order=spline_order,
                                  reflect_range=post_reflect_range, refract_range=post_refract_range,
                                  deriv=post_deriv, deriv_func=deriv_func, **post_process_args)

    return tensor