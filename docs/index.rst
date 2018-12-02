Noisemaker
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

**Noisemaker** is an adaptation of classic procedural noise generation algorithms, for `Python 3`_ and `TensorFlow`_.

.. image:: https://travis-ci.com/aayars/py-noisemaker.svg?branch=master
   :target: https://travis-ci.com/aayars/py-noisemaker
   :alt: Build Status

.. image:: https://readthedocs.org/projects/noisemaker/badge/?version=latest
   :target: https://noisemaker.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/docker/build/aayars/py-noisemaker.svg
   :target: https://hub.docker.com/r/aayars/py-noisemaker
   :alt: Docker Status

Installation
------------

Python 3 virtualenv
~~~~~~~~~~~~~~~~~~~

Noisemaker is intended for Python 3.5+.

Install `Noisemaker`_ in a new virtualenv:

.. code-block:: bash

    python3 -m venv noisemaker

    source noisemaker/bin/activate

    pip install git+https://github.com/aayars/py-noisemaker

    pip install tensorflow  # tensorflow==1.3.0 recommended, tensorflow-gpu requires cuda/cudnn

For subsequent activation of the virtual environment, run `source bin/activate` while in the `noisemaker` directory. To deactivate, run `deactivate`.

Upgrading
~~~~~~~~~

Activate the virtual environment, and run:

.. code-block:: bash

    pip install --upgrade git+https://github.com/aayars/py-noisemaker

Development
~~~~~~~~~~~

To install noisemaker in a dev env:

.. code-block:: bash

    git clone https://github.com/aayars/py-noisemaker

    cd py-noisemaker

    python3 -m venv venv

    source venv/bin/activate

    python setup.py develop
    python setup.py install_scripts

For subsequent activation of the virtual environment, run `source venv/bin/activate` while in the `noisemaker` directory. To deactivate, run `deactivate`.

Docker
~~~~~~

Noisemaker can run on CPU in a container. See `Noisemaker on Docker`_!

Usage
-----

API
~~~

See :doc:`api` documentation.

CLI
~~~

Noisemaker includes several CLI entrypoints. For usage summary, run with `-h` or `--help`.

-  `noisemaker`: Fully-featured noise generation pipeline
-  `artmaker`: Generate images from presets
-  `artmangler`: Apply post-processing effects to images

See also
--------

-  `Wikipedia: Value noise`_
-  `Wikipedia: Perlin noise`_
-  `Wikipedia: Voronoi diagram`_
-  `Wikipedia: Worley noise`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. _`Python 3`: https://www.python.org/
.. _`Noisemaker`: https://github.com/aayars/py-noisemaker
.. _`Noisemaker on Docker`: https://github.com/aayars/py-noisemaker/blob/master/docker/README.md
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`install TensorFlow`: https://www.tensorflow.org/install/
.. _`Wikipedia: Value noise`: https://en.wikipedia.org/wiki/Value_noise
.. _`Wikipedia: Perlin noise`: https://en.wikipedia.org/wiki/Perlin_noise
.. _`Wikipedia: Voronoi diagram`: https://en.wikipedia.org/wiki/Voronoi_diagram
.. _`Wikipedia: Worley noise`: https://en.wikipedia.org/wiki/Worley_noise
