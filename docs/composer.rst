Noisemaker Composer
===================

Noisemaker Composer is a high-level interface for creating generative art with noise. The design is informed by lessons learned
from previous preset systems in Noisemaker.

The modern preset library is authored with the preset DSL described in :file:`PRESET_DSL_SPEC.md`. The canonical source lives in
:file:`dsl/presets.dsl` and is parsed by :mod:`noisemaker.dsl`. Both the reference Python implementation and the JavaScript port
consume that file, so any change to the DSL immediately applies to both environments.

Composer Presets
----------------

"Composer" presets expose Noisemaker's lower-level `generator <api.html#module-noisemaker.generators>`_ and
`effect <api.html#module-noisemaker.effects>`_ APIs. The DSL keeps the data model compact while remaining expressive enough to
describe the same building blocks the original Python dictionaries exposed. At a high level each preset still answers five key
questions:

1) Which presets are being built on?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Effects and presets are intended to be combined with and riff off each other, but repetition in code is distasteful, especially
when we have to copy settings around. To minimize copied settings, presets may ``layer`` parent presets and/or reference other
presets inline.

Layering in this way means inheriting from those presets as a starting point without copying everything in. A preset with no
layers defined starts from a blank slate using default generator parameters and no effects.

The lineage of ancestor presets is modeled in each preset's ``layers`` list, which is a flat list of preset names in the order
they should be applied.

.. code-block:: javascript

    "just-an-example": {
      // A list of parent preset names, if any:
      layers: ["first-parent", "second-parent"],
    }

2) What are the meaningful variables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each preset may need to reuse values or tweak a value set by a parent. The ``settings`` dictionary provides an overridable bank
of variables. Layering inherits this dictionary, allowing preset authors to override or add key/value pairs. Values may be
literal, reference enums, call helper functions like ``random_int`` or ``coin_flip``, or defer evaluation by wrapping work in a
callable. Within the DSL, preset authors access previously defined settings with the ``settings.`` prefix.

.. code-block:: javascript

    "just-an-example": {
      settings: {
        your_special_variable: random(),
        another_special_variable: random_int(2, 4),
      },
    }

3) What are the noise generation parameters?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Noise generator parameters are modeled in each preset's ``generator`` dictionary. Generator parameters can be literal values or
resolve values from settings. Just as with ``settings``, layering inherits this dictionary. Unlike ``settings``, the keys in
``generator`` must be valid parameters to `noisemaker.generators.multires <api.html#noisemaker.generators.multires>`_.

.. code-block:: javascript

    "just-an-example": {
      generator: {
        freq: settings.base_freq,
        octaves: random_int(4, 8),
        ridges: true,
      },
    }

4) Which effects should be applied to each octave?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preset authors may specify an arbitrary list of effects to apply per octave of noise. Per-octave effects are modeled in each
preset's ``octaves`` list, which contains parameterized effect calls. Effect parameters may be defined inline or fed in from
settings. Layering inherits this list, allowing authors to append additional effects. Effects are listed in the order to be
applied.

.. code-block:: javascript

    "just-an-example": {
      octaves: [
        derivative(alpha: settings.deriv_alpha),
        ripple(range: 0.05),
      ],
    }

5) Which effects should be applied after flattening layers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-reduce effects are modeled in each preset's ``post`` list, and final pass effects live in ``final``. Both sections accept
parameterized effect calls as well as ``preset("name")`` references that inline the referenced preset's ``post``/``final``
steps. Layering inherits these lists, enabling preset authors to append additional effects and nested presets. Effects are
applied in order.

.. code-block:: javascript

    "just-an-example": {
      post: [
        bloom(alpha: settings.bloom_alpha),
        preset("vignette"),
      ],
      final: [
        adjust_contrast(amount: 1.1),
      ],
    }

Putting It All Together
-----------------------

The following contrived example illustrates a preset containing each of the above described sections. For concrete examples, see
:file:`dsl/presets.dsl`, :mod:`noisemaker.presets`, and :mod:`test.test_composer`.

.. code-block:: javascript

    {
      "just-an-example": {
        layers: ["first-parent", "second-parent"],

        settings: {
          base_freq: random_int(2, 4),
          bloom_alpha: 0.1 + random() * 0.05,
        },

        generator: {
          freq: settings.base_freq,
          octaves: 6,
          ridges: true,
        },

        octaves: [
          derivative(alpha: 0.333),
          ripple(range: 0.05),
        ],

        post: [
          bloom(alpha: settings.bloom_alpha),
          preset("grain"),
        ],

        final: [
          adjust_contrast(amount: 1.1),
        ],
      },
    }
