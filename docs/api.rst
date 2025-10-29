Noisemaker API
==============

Images are float32 tensors (0..1 range), with shape (height, width, channels). Most functions assume seamlessly tiled noise.

This is a pre-1.0 API, and may receive backwards incompatible changes.

.. note::
   **JavaScript Port Available**: Noisemaker includes a vanilla JavaScript port with WebGPU acceleration. 
   See :doc:`javascript` for the browser-based API that mirrors this Python API.

noisemaker.generators
---------------------

High-level noise generation functions.

.. automodule:: noisemaker.generators
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.composer
-------------------

Preset-based noise composition system.

.. automodule:: noisemaker.composer
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.value
----------------

Low-level value noise functions.

.. automodule:: noisemaker.value
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.effects
------------------

Image effect and post-processing functions.

.. automodule:: noisemaker.effects
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.util
---------------

Utility functions for I/O and image processing.

.. automodule:: noisemaker.util
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.rng
--------------

Deterministic random number generation.

.. automodule:: noisemaker.rng
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.oklab
----------------

Oklab color space conversion utilities.

.. automodule:: noisemaker.oklab
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.constants
--------------------

Enumeration constants for noise generation.

.. automodule:: noisemaker.constants
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.masks
----------------

Mask generation and application functions.

The ``Masks`` dictionary contains pre-defined mask patterns for all ValueMask enum members.
See :class:`noisemaker.constants.ValueMask` for available mask types.

.. automodule:: noisemaker.masks
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: Masks

noisemaker.points
-----------------

Point cloud generation utilities.

.. automodule:: noisemaker.points
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.palettes
-------------------

Color palette definitions and utilities.

.. automodule:: noisemaker.palettes
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.simplex
------------------

Simplex noise implementation.

.. automodule:: noisemaker.simplex
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.glyphs
-----------------

Font and glyph rendering utilities.

.. automodule:: noisemaker.glyphs
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.presets
------------------

Preset loading and management.

.. automodule:: noisemaker.presets
    :members:
    :undoc-members:
    :show-inheritance:

noisemaker.effects_registry
---------------------------

Effect function registry and decorators.

.. automodule:: noisemaker.effects_registry
    :members:
    :undoc-members:
    :show-inheritance:
