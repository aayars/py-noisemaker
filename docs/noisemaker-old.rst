Noisemaker CLI (Old)
====================

Noisemaker includes a now-deprecated low-level CLI utility, ``noisemaker-old``.

This utility exposes many of the possible noise options and is not for the faint of heart. If you're seeking a simpler noise 
generation experience, take a look at the newer :doc:`noisemaker` instead.

Sample usage:

.. code-block:: bash

    noisemaker-old --shadow 0.1 --ridges --width 256 --height 256

    # Maybe it's ugly, maybe it's awesome. Random values are random!

.. image:: images/noise.jpg

See ``noisemaker-old --help`` for a complete list of options.

.. literalinclude:: noisemaker-old-help.txt
   :language: bash
