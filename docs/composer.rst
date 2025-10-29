Noisemaker Composer DSL
=======================

Noisemaker Composer is a high-level interface for creating generative art with procedural noise. The design is informed by lessons learned from previous preset systems in Noisemaker.

The modern preset library is authored with the **Composer DSL**, a domain-specific language for defining procedural art presets. It provides a structured, declarative syntax for composing noise generators and image effects while supporting randomization, inheritance, and reusable configurations.

.. note::
   **JavaScript Compatibility**: The Composer DSL is shared between Python and JavaScript implementations.
   See :doc:`javascript` for using presets in the browser with WebGPU.

Overview
--------

The Composer DSL allows you to define presets as JSON-like objects with specialized syntax extensions. Each preset can:

* Layer (inherit from) other presets
* Define reusable settings with random values
* Configure noise generation parameters
* Apply effects per-octave, post-processing, and final passes
* Reference other presets inline

The canonical preset library lives in :file:`dsl/presets.dsl` and is parsed by :mod:`noisemaker.dsl`. Both the reference Python implementation and the JavaScript port consume that file, so any change to the DSL immediately applies to both environments.

Philosophy
~~~~~~~~~~

"Composer" presets expose Noisemaker's lower-level `generator <api.html#module-noisemaker.generators>`_ and `effect <api.html#module-noisemaker.effects>`_ APIs. The DSL keeps the data model compact while remaining expressive enough to describe the same building blocks the original Python dictionaries exposed. 

At a high level, each preset answers five key questions:

1. Which presets are being built on? (``layers``)
2. What are the meaningful variables? (``settings``)
3. What are the noise generation parameters? (``generator``)
4. Which effects should be applied to each octave? (``octaves``)
5. Which effects should be applied after flattening layers? (``post`` and ``final``)

Basic Structure
---------------

A minimal preset looks like this:

.. code-block:: javascript

    {
      "my-preset": {
        settings: {
          my_freq: random_int(2, 4),
        },
        generator: {
          freq: settings.my_freq,
        },
      },
    }

Preset Keys
-----------

Each preset is a dictionary that can contain the following top-level keys:

layers
~~~~~~

**Type:** Array of strings

**Purpose:** Inherit settings, generators, and effects from parent presets.

Parent presets are applied in order, with later presets overriding earlier ones. The current preset's settings override all parents.

.. code-block:: javascript

    layers: ["basic", "voronoi", "grain"]

settings
~~~~~~~~

**Type:** Dictionary

**Purpose:** Define variables that can be referenced throughout the preset using ``settings.key_name``.

Settings support:

* Literal values (numbers, strings, booleans, null)
* Enum references
* Helper function calls (``random()``, ``random_int()``, ``coin_flip()``, etc.)
* Arithmetic expressions
* Conditional expressions (ternary)

.. code-block:: javascript

    settings: {
      base_freq: random_int(2, 8),
      use_ridges: coin_flip(),
      hue_value: random(),
      computed: settings.base_freq * 2,
    }

generator
~~~~~~~~~

**Type:** Dictionary

**Purpose:** Configure noise generation parameters passed to ``noisemaker.generators.multires``.

All keys must be valid generator parameters. Values can reference settings or be literals.

.. code-block:: javascript

    generator: {
      freq: settings.base_freq,
      octaves: 6,
      ridges: settings.use_ridges,
      lattice_drift: 0.5,
    }

Common generator parameters:

* ``freq`` - Frequency of noise (can be int or [width, height])
* ``octaves`` - Number of octaves for multi-resolution noise
* ``ridges`` - Enable ridge noise (boolean)
* ``distrib`` - Value distribution (ValueDistribution enum)
* ``color_space`` - Color space (ColorSpace enum)
* ``hue_range`` - Hue variation range (0.0-1.0)
* ``lattice_drift`` - Lattice drift amount
* ``corners`` - Enable corner artifacts (boolean)
* ``spline_order`` - Interpolation type (InterpolationType enum)

octaves
~~~~~~~

**Type:** Array of effect calls

**Purpose:** Effects applied to each octave of noise during generation.

.. code-block:: javascript

    octaves: [
      derivative(alpha: 0.5),
      ripple(range: 0.1),
    ]

post
~~~~

**Type:** Array of effect calls and/or preset references

**Purpose:** Effects applied after noise octaves are combined.

.. code-block:: javascript

    post: [
      bloom(alpha: 0.25),
      preset("vignette"),
      saturation(amount: 1.5),
    ]

final
~~~~~

**Type:** Array of effect calls and/or preset references

**Purpose:** Final effects applied after all post-processing.

.. code-block:: javascript

    final: [
      aberration(displacement: 0.01),
      adjust_contrast(amount: 1.1),
    ]

unique
~~~~~~

**Type:** Boolean

**Purpose:** Mark preset as unique (not for general layering). Defaults to false.

.. code-block:: javascript

    unique: true

Data Types
----------

The DSL supports the following data types:

Numbers
~~~~~~~

Integers and floats, including arithmetic expressions:

.. code-block:: javascript

    freq: 5
    alpha: 0.5 + random() * 0.25
    computed: settings.base * 2 + 1

Strings
~~~~~~~

Double-quoted strings (no escape sequences):

.. code-block:: javascript

    palette_name: "viridis"

Booleans
~~~~~~~~

Keywords ``true`` and ``false``:

.. code-block:: javascript

    ridges: true
    inverse: false

Null
~~~~

Keyword ``null``:

.. code-block:: javascript

    mask: null

Arrays
~~~~~~

Lists of values:

.. code-block:: javascript

    freq: [4, 8]
    layers: ["basic", "grain"]
    options: [1, 2, 3]

Dictionaries
~~~~~~~~~~~~

Key-value pairs:

.. code-block:: javascript

    settings: {
      key1: value1,
      key2: value2,
    }

Enums
~~~~~

Access enum members using dot notation:

.. code-block:: javascript

    color_space: ColorSpace.rgb
    mask: ValueMask.chess
    dist_metric: DistanceMetric.euclidean

Available enums include: ``ColorSpace``, ``ValueDistribution``, ``ValueMask``, ``DistanceMetric``, ``VoronoiDiagramType``, ``PointDistribution``, ``InterpolationType``, ``OctaveBlending``, ``WormBehavior``, and more.

Expressions
-----------

Arithmetic
~~~~~~~~~~

Standard operators: ``+``, ``-``, ``*``, ``/``

.. code-block:: javascript

    value: 0.5 + random() * 0.25
    doubled: settings.freq * 2
    averaged: (settings.a + settings.b) / 2

Conditional (Ternary)
~~~~~~~~~~~~~~~~~~~~~

JavaScript-style ternary:

.. code-block:: javascript

    value: coin_flip() ? 1 : 0
    freq: random() < 0.5 ? 4 : 8

Python-style conditional:

.. code-block:: javascript

    value: 1 if coin_flip() else 0

Comparison and Logic
~~~~~~~~~~~~~~~~~~~~

Comparison operators: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``

Logical operators: ``&&`` (and), ``||`` (or)

.. code-block:: javascript

    use_effect: random() < 0.75
    value: (settings.a > 10 && settings.b < 5) ? 1 : 0

Settings References
~~~~~~~~~~~~~~~~~~~

Access previously defined settings:

.. code-block:: javascript

    settings: {
      base_freq: random_int(2, 8),
      double_freq: settings.base_freq * 2,
      derived: settings.base_freq + settings.double_freq,
    }

Helper Functions
----------------

The DSL provides built-in helper functions for randomization and utilities:

random()
~~~~~~~~

Returns a random float between 0.0 and 1.0.

.. code-block:: javascript

    alpha: 0.5 + random() * 0.5

**RNG Impact:** Consumes 1 random number from the generator.

random_int(min, max)
~~~~~~~~~~~~~~~~~~~~

Returns a random integer between ``min`` and ``max`` (inclusive).

.. code-block:: javascript

    freq: random_int(2, 8)
    octaves: random_int(4, 12)

**RNG Impact:** Consumes 1 random number from the generator.

coin_flip()
~~~~~~~~~~~

Returns a random boolean (true or false).

.. code-block:: javascript

    ridges: coin_flip()
    should_invert: coin_flip()

**RNG Impact:** Consumes 1 random number from the generator.

random_member(collection, ...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a random member from one or more collections. Multiple collections are flattened and sorted deterministically before selection.

.. code-block:: javascript

    // From array
    value: random_member([1, 2, 3])
    
    // From enum
    color_space: random_member(ColorSpace.color_members())
    
    // Multiple collections
    option: random_member([1, 2], [3, 4])
    
    // Multiple enums
    dist: random_member(
      DistanceMetric.absolute_members(),
      [DistanceMetric.euclidean]
    )

**RNG Impact:** Consumes 1 random number from the generator.

enum_range(start, end)
~~~~~~~~~~~~~~~~~~~~~~

Returns a list of integers from ``start`` to ``end`` (inclusive).

.. code-block:: javascript

    values: enum_range(1, 5)  // [1, 2, 3, 4, 5]

**RNG Impact:** None (deterministic).

stash(key, value)
~~~~~~~~~~~~~~~~~

Stores a value for later retrieval within the same evaluation context.

.. code-block:: javascript

    // Store
    temp: stash("my_key", 42)
    
    // Retrieve
    retrieved: stash("my_key")

**RNG Impact:** None.

mask_freq(mask, repeat)
~~~~~~~~~~~~~~~~~~~~~~~

Returns the appropriate frequency for a given mask and repeat value.

.. code-block:: javascript

    freq: mask_freq(ValueMask.chess, 8)

**RNG Impact:** None.

preset(name)
~~~~~~~~~~~~

Inline another preset's post/final effects.

.. code-block:: javascript

    post: [
      bloom(alpha: 0.25),
      preset("vignette"),  // Includes vignette's post/final
    ]

**RNG Impact:** Depends on the referenced preset.

Enum Helper Methods
-------------------

Enums provide helper methods to filter and retrieve specific members:

``EnumType.all()``
~~~~~~~~~~~~~~~~~~

Returns all enum members.

.. code-block:: javascript

    all_colors: ColorSpace.all()

Specific Enum Methods
~~~~~~~~~~~~~~~~~~~~~

Different enums provide specialized filter methods:

**ColorSpace:**

* ``ColorSpace.color_members()`` - Color spaces only

**DistanceMetric:**

* ``DistanceMetric.absolute_members()`` - Absolute metrics
* ``DistanceMetric.all()`` - All metrics

**ValueMask:**

* ``ValueMask.procedural_members()`` - Procedural masks
* ``ValueMask.grid_members()`` - Grid-based masks
* ``ValueMask.glyph_members()`` - Glyph/character masks
* ``ValueMask.nonprocedural_members()`` - Non-procedural masks
* ``ValueMask.rgb_members()`` - RGB-based masks

**PointDistribution:**

* ``PointDistribution.circular_members()`` - Circular distributions
* ``PointDistribution.grid_members()`` - Grid-based distributions

**WormBehavior:**

* ``WormBehavior.all()`` - All worm behaviors

Example usage:

.. code-block:: javascript

    settings: {
      color_space: random_member(ColorSpace.color_members()),
      mask: random_member(ValueMask.grid_members()),
      dist: random_member(DistanceMetric.absolute_members()),
    }

Effect Calls
------------

Effects are called with named parameters using colon syntax:

.. code-block:: javascript

    effect_name(param1: value1, param2: value2)

Examples:

.. code-block:: javascript

    octaves: [
      derivative(alpha: 0.5),
      ripple(range: 0.1, freq: 2),
    ]
    
    post: [
      bloom(alpha: settings.bloom_alpha),
      saturation(amount: 1.5),
      rotate(angle: settings.rotation),
    ]

Common effects and their parameters:

**Color/Hue:**

* ``random_hue()`` - Randomize hue
* ``nudge_hue(amount)`` - Slight hue shift
* ``saturation(amount)`` - Adjust saturation

**Blur/Bloom:**

* ``bloom(alpha)`` - Bloom/glow effect
* ``vaseline(alpha)`` - Blur effect

**Distortion:**

* ``aberration(displacement)`` - Chromatic aberration
* ``ripple(range, freq)`` - Ripple distortion
* ``warp(displacement, octaves, freq)`` - Warp effect
* ``funhouse()`` - Funhouse mirror effect

**Tone/Contrast:**

* ``adjust_contrast(amount)`` - Adjust contrast
* ``normalize()`` - Normalize values
* ``posterize(levels)`` - Posterize colors
* ``vignette(alpha, brightness)`` - Vignette effect

**Texture:**

* ``grain()`` - Add film grain
* ``snow(alpha)`` - Add noise
* ``spatter(amount)`` - Spatter effect

**Geometry:**

* ``rotate(angle)`` - Rotate image
* ``reflect(orientation)`` - Mirror/reflect
* ``symmetry()`` - Create symmetry

See the `effects API documentation <api.html#module-noisemaker.effects>`_ for complete parameter lists.

Complete Example
----------------

Here's a complete preset demonstrating all major features:

.. code-block:: javascript

    {
      "example-preset": {
        // Inherit from parent presets
        layers: ["basic", "voronoi"],
        
        // Define reusable settings
        settings: {
          // Random values
          base_freq: random_int(4, 8),
          bloom_alpha: 0.1 + random() * 0.15,
          use_ridges: coin_flip(),
          
          // Conditional values
          octave_count: random() < 0.5 ? 4 : 8,
          
          // Enum selection
          color_space: random_member(ColorSpace.color_members()),
          
          // Computed values
          double_freq: settings.base_freq * 2,
        },
        
        // Configure noise generation
        generator: {
          freq: settings.base_freq,
          octaves: settings.octave_count,
          ridges: settings.use_ridges,
          color_space: settings.color_space,
          distrib: ValueDistribution.simplex,
        },
        
        // Per-octave effects
        octaves: [
          derivative(alpha: 0.333),
        ],
        
        // Post-processing effects
        post: [
          bloom(alpha: settings.bloom_alpha),
          preset("grain"),  // Inline another preset
          saturation(amount: 1.25),
        ],
        
        // Final pass effects
        final: [
          aberration(displacement: 0.0125),
          adjust_contrast(amount: 1.1),
        ],
      },
    }

Naming Conventions
------------------

The DSL follows these naming conventions:

* **Preset names**: ``kebab-case`` (e.g., ``"my-awesome-preset"``)
* **Setting keys**: ``snake_case`` (e.g., ``base_freq``, ``bloom_alpha``)
* **Function names**: ``snake_case`` (e.g., ``random_int``, ``coin_flip``)
* **Enum types**: ``PascalCase`` (e.g., ``ColorSpace``, ``ValueMask``)
* **Enum members**: ``snake_case`` (e.g., ``ColorSpace.rgb``, ``ValueMask.chess``)

Best Practices
--------------

1. **Use settings for reusable values**

   Store commonly used values in ``settings`` to avoid repetition and make presets easier to tune:

   .. code-block:: javascript

       settings: {
         bloom_alpha: 0.25,
       },
       post: [
         bloom(alpha: settings.bloom_alpha),
       ]

2. **Layer presets for composition**

   Build complex presets by layering simpler ones:

   .. code-block:: javascript

       layers: ["basic", "grain", "saturation"]

3. **Use descriptive setting names**

   Make your intent clear:

   .. code-block:: javascript

       settings: {
         vignette_brightness: 0.5,  // Good
         vb: 0.5,                     // Bad
       }

4. **Understand RNG consumption**

   Be aware that ``random()``, ``random_int()``, ``coin_flip()``, and ``random_member()`` all advance the random number generator. The order of evaluation matters for reproducible results.

5. **Use conditional effects**

   Make presets more varied by conditionally including effects:

   .. code-block:: javascript

       post: coin_flip() ? [bloom(alpha: 0.25)] : []

6. **Reference the canonical library**

   Study existing presets in :file:`dsl/presets.dsl` for patterns and techniques.

Debugging
---------

When a preset doesn't parse or evaluate correctly:

1. **Check syntax**: Ensure all braces, brackets, and parentheses are balanced
2. **Verify enum names**: Enum references must exactly match defined enums
3. **Check parameter names**: Effect parameters must match the effect's signature
4. **Look for typos**: Setting references must exactly match defined keys
5. **Test incrementally**: Build complex presets step-by-step

The Python and JavaScript parsers provide error messages with line/column information when syntax errors occur.

Using Presets in Python
------------------------

The Composer API provides high-level access to presets defined in the DSL.

Basic Usage
~~~~~~~~~~~

Import and instantiate a preset by name:

.. code-block:: python

    from noisemaker.composer import Preset

    preset = Preset('acid')
    # Render directly to a file
    preset.render(seed=1, shape=[256, 256, 3], filename='art.png')

The ``shape`` parameter defines the output dimensions as ``[height, width, channels]``. Use 3 channels for RGB color images.

Working with Arrays
~~~~~~~~~~~~~~~~~~~

To work with the generated data as a NumPy array instead of writing to a file:

.. code-block:: python

    from noisemaker.composer import Preset

    preset = Preset('acid')
    # Returns a TensorFlow tensor
    tensor = preset.render(seed=1, shape=[256, 256, 3])
    # Convert to NumPy array
    array = tensor.numpy()

Custom Settings
~~~~~~~~~~~~~~~

Override preset settings at render time:

.. code-block:: python

    from noisemaker.composer import Preset

    preset = Preset('acid', settings={'freq': 20, 'octaves': 12})
    preset.render(seed=1, shape=[256, 256, 3], filename='custom-acid.png')

This allows you to tweak parameters without modifying the DSL file.

Available Presets
~~~~~~~~~~~~~~~~~

List all available presets:

.. code-block:: python

    from noisemaker.presets import PRESETS

    presets = PRESETS()
    print(list(presets.keys()))

Or explore the canonical DSL file at :file:`dsl/presets.dsl`.

Architecture Overview
---------------------

The Noisemaker Composer system is built on three layers:

1. **DSL Layer** (:mod:`noisemaker.dsl`)
   
   Parses and evaluates the Composer DSL from :file:`dsl/presets.dsl`. The same DSL file is used by both Python and JavaScript implementations, ensuring cross-platform consistency.

2. **Preset Layer** (:mod:`noisemaker.presets`, :mod:`noisemaker.composer`)
   
   Loads preset definitions and provides the ``Preset`` class for rendering. Handles preset inheritance (layering), settings resolution, and effect application.

3. **Generator/Effect Layer** (:mod:`noisemaker.generators`, :mod:`noisemaker.effects`)
   
   Low-level TensorFlow operations for generating procedural noise and applying image effects.

The DSL provides a declarative interface to these lower-level APIs, making it easy to compose complex generative art without writing imperative code.

Cross-Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the Python and JavaScript implementations:

* Parse the same DSL file (:file:`dsl/presets.dsl`)
* Use identical tokenizer, parser, and evaluator logic
* Produce deterministic output given the same seed
* Support the same set of helper functions and enums

Any change to the DSL immediately applies to both environments, making it easy to maintain consistency across platforms.

See Also
--------

* :doc:`api` - Low-level generator and effect APIs
* :doc:`cli` - Command-line interface documentation
* :mod:`noisemaker.presets` - Preset loading and evaluation
* :mod:`noisemaker.dsl` - DSL parser and evaluator modules
* :mod:`noisemaker.composer` - Composer class and rendering API
