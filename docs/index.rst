.. Noisemaker documentation master file, created by
   sphinx-quickstart on Mon Nov 30 19:51:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Noisemaker
==========

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   composer
   api
   cli
   javascript


**Noisemaker** is an adaptation of classic procedural noise generation algorithms, for `Python 3.9+`_ and `TensorFlow`_.

**Noisemaker also includes a vanilla JavaScript port** that runs in modern browsers with WebGPU support. See the :doc:`javascript` documentation for details.

Installation
------------

Python 3.9+ virtualenv
~~~~~~~~~~~~~~~~~~~~~~

Noisemaker requires Python 3.9 or later.

Install `Noisemaker`_ in a new virtualenv:

.. code-block:: bash

    python3 -m venv noisemaker

    source noisemaker/bin/activate

    pip install git+https://github.com/aayars/py-noisemaker

TensorFlow is included as a dependency and will be installed automatically.

For subsequent activation of the virtual environment, run ``source bin/activate`` while in the ``noisemaker`` directory. To deactivate, run ``deactivate``.

Upgrading
~~~~~~~~~

Activate the virtual environment, and run:

.. code-block:: bash

    pip install --upgrade git+https://github.com/aayars/py-noisemaker

Development
~~~~~~~~~~~

To install noisemaker in a dev env with code quality tools:

.. code-block:: bash

    git clone https://github.com/aayars/py-noisemaker

    cd py-noisemaker

    python3 -m venv venv

    source venv/bin/activate

    pip install -e ".[dev]"

    pre-commit install

This installs noisemaker with development dependencies including black, ruff, mypy, and pytest.

To run tests:

.. code-block:: bash

    pytest

To format and lint code:

.. code-block:: bash

    black noisemaker
    ruff check noisemaker
    mypy noisemaker

For subsequent activation of the virtual environment, run ``source venv/bin/activate`` while in the ``noisemaker`` directory. To deactivate, run ``deactivate``.

Notebook
~~~~~~~~

You can play with Noisemaker in a `Colab Notebook`_.

Docker
~~~~~~

Noisemaker can run on CPU in a container. See `Noisemaker on Docker`_!

Usage
-----

CLI
~~~

See :doc:`cli` documentation.

High-level API: Noisemaker Composer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`composer` documentation.

Low-level API: Generator and Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`api` documentation.


See also
--------

-  `Wikipedia: Value noise`_
-  `Wikipedia: Voronoi diagram`_
-  `Wikipedia: Worley noise`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`Python 3.9+`: https://www.python.org/
.. _`Python 3`: https://www.python.org/
.. _`Noisemaker`: https://github.com/aayars/py-noisemaker
.. _`Colab Notebook`: https://colab.research.google.com/github/aayars/py-noisemaker/blob/master/py_noisemaker.ipynb
.. _`Noisemaker on Docker`: https://hub.docker.com/r/aayars/py-noisemaker/
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`install TensorFlow`: https://www.tensorflow.org/install/
.. _`Wikipedia: Value noise`: https://en.wikipedia.org/wiki/Value_noise
.. _`Wikipedia: Voronoi diagram`: https://en.wikipedia.org/wiki/Voronoi_diagram
.. _`Wikipedia: Worley noise`: https://en.wikipedia.org/wiki/Worley_noise

