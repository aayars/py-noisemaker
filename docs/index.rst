Noisemaker
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: images/sploosh.jpg
   :width: 1024
   :height: 256
   :alt: Noisemaker example output (CC0)

**Noisemaker** is a visual noise generator for `Python 3`_.

Installation
------------

Noisemaker is intended for Python 3.5+.

Install `Noisemaker`_ in a new virtualenv:

.. code-block:: bash

    python3 -m venv noisemaker

    source noisemaker/bin/activate

    pip install git+https://github.com/aayars/py-noisemaker

`Install TensorFlow`_ in the virtualenv, using ``pip``. See TensorFlow's platform-specific docs for your ``$TF_BINARY_URL``.

.. code-block:: bash

    # Try to see if there is a wheel available for your platform.
    pip install tensorflow  # or tensorflow-gpu, if you're all set up with cuda/cudnn

    # pip install $TF_BINARY_URL
    # e.g. Mac OS X CPU:
    # https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl

Usage
-----

See documentation for :doc:`api` or :doc:`cli`: ``noisemaker --help``

See also
--------

-  `Wikipedia: Value Noise`_
-  `Wikipedia: Perlin Noise`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _`Python 3`: https://www.python.org/
.. _`Noisemaker`: https://github.com/aayars/py-noisemaker
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`install TensorFlow`: https://www.tensorflow.org/install/
.. _`Wikipedia: Value Noise`: https://en.wikipedia.org/wiki/Value_noise
.. _`Wikipedia: Perlin Noise`: https://en.wikipedia.org/wiki/Perlin_noise
