Noisemaker
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: images/sploosh.jpg
   :width: 1024
   :height: 256
   :alt: Noisemaker example output (CC0)

**Noisemaker** is an adaptation of classic procedural noise generation algorithms, for `Python 3` and `TensorFlow`.

Installation
------------

Docker
~~~~~~

Noisemaker can run on CPU in a container. See `Noisemaker on Docker`_.

Not Docker (Python 3 virtualenv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Noisemaker is intended for Python 3.5+.

Install `Noisemaker`_ in a new virtualenv:

.. code-block:: bash

    python3 -m venv noisemaker

    source noisemaker/bin/activate

    pip install git+https://github.com/aayars/py-noisemaker

`Install TensorFlow`_ in the virtualenv, using ``pip``.

.. code-block:: bash

    # Hopefully there is a wheel available for your platform.
    pip install tensorflow  # or tensorflow-gpu, if you're all set up with cuda/cudnn

    # pip install $TF_BINARY_URL
    # e.g. Mac OS X CPU:
    # https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py3-none-any.whl

Upgrading
^^^^^^^^^

Activate the virtual environment, and run:

    pip install --upgrade git+https://github.com/aayars/py-noisemaker

Usage
-----

API
~~~

See :doc:`api` documentation.

CLI
~~~

Noisemaker includes several CLI entrypoints. For usage summary, run with `-h` or `--help`.

-  `noisemaker`: Fully-featured noise generation pipeline
-  `glitchmaker`: Simple glitch art tool
-  `collagemaker`: Image collage tool

See also
--------

-  `Wikipedia: Value Noise`_
-  `Wikipedia: Perlin Noise`_
-  `Wikipedia: Voronoi diagram`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _`Python 3`: https://www.python.org/
.. _`Noisemaker`: https://github.com/aayars/py-noisemaker
.. _`Noisemaker on Docker`: https://github.com/aayars/py-noisemaker/blob/master/docker/README.md
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`install TensorFlow`: https://www.tensorflow.org/install/
.. _`Wikipedia: Value Noise`: https://en.wikipedia.org/wiki/Value_noise
.. _`Wikipedia: Perlin Noise`: https://en.wikipedia.org/wiki/Perlin_noise
.. _`Wikipedia: Voronoi diagram`: https://en.wikipedia.org/wiki/Voronoi_diagram