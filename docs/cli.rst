Noisemaker CLI
==============

The ``noisemaker`` command-line interface provides tools for generating procedural art, creating animations, and applying effects to images using `Composer Presets <composer.html>`_.

Overview
--------

The CLI includes five main commands:

- ``generate`` - Create a single image from a preset
- ``animate`` - Create an animation from a preset
- ``apply`` - Apply an effect preset to an existing image
- ``mashup`` - Blend multiple images together
- ``magic-mashup`` - Create animated collages from directories of frames

Quick Start
-----------

Generate a simple noise image:

.. code-block:: bash

    noisemaker generate basic -o output.png

Create an animation:

.. code-block:: bash

    noisemaker animate acid --width 1024 --height 1024 -o acid.mp4

Apply an effect to an existing image:

.. code-block:: bash

    noisemaker apply glitchin-out input.jpg -o glitched.png

Commands
--------

generate
~~~~~~~~

Generate a .png or .jpg image from a preset.

.. code-block:: bash

    noisemaker generate PRESET_NAME [OPTIONS]

**Required Arguments:**

- ``PRESET_NAME`` - Name of the preset to use (e.g., ``acid``, ``voronoi``, ``multires``)

**Common Options:**

- ``--width INTEGER`` - Output width in pixels (default: 1024)
- ``--height INTEGER`` - Output height in pixels (default: 1024)
- ``--seed INTEGER`` - Random seed for reproducible results
- ``--filename FILE`` - Output filename (default: art.png)
- ``--time FLOAT`` - Time value for Z axis (for 3D simplex noise)
- ``--speed FLOAT`` - Animation speed modifier

**Quality Options:**

- ``--with-alpha`` - Include alpha channel in output
- ``--with-supersample`` - Apply 2x supersample anti-aliasing
- ``--with-fxaa`` - Apply FXAA anti-aliasing

**AI Features (requires API keys):**

- ``--with-ai`` - Apply image-to-image transformation (requires stability.ai key)
- ``--with-upscale`` - Apply 4x upscaling (requires stability.ai key)
- ``--with-alt-text`` - Generate alt text description (requires OpenAI key)
- ``--stability-model TEXT`` - Override default stability.ai model

**Debug Options:**

- ``--debug-print`` - Print preset ancestry and settings to stdout
- ``--debug-out FILE`` - Write preset ancestry and settings to file

**Examples:**

.. code-block:: bash

    # Generate a basic image with custom dimensions
    noisemaker generate multires --width 2048 --height 2048 -o noise.png

    # Generate with a specific seed for reproducibility
    noisemaker generate acid --seed 12345 -o acid.png

    # Generate with anti-aliasing
    noisemaker generate voronoi --with-fxaa -o smooth.png

    # Generate and apply AI upscaling
    noisemaker generate fractal-smoke --with-upscale -o hires.png

animate
~~~~~~~

Generate an animation (MP4 or GIF) from a preset.

.. code-block:: bash

    noisemaker animate PRESET_NAME [OPTIONS]

**Required Arguments:**

- ``PRESET_NAME`` - Name of the preset to animate

**Common Options:**

- ``--width INTEGER`` - Output width in pixels (default: 512)
- ``--height INTEGER`` - Output height in pixels (default: 512)
- ``--filename FILE`` - Output filename (default: animation.mp4)
- ``--frame-count INTEGER`` - Number of frames to generate (default: 50)
- ``--seed INTEGER`` - Random seed for reproducible results
- ``--effect-preset NAME`` - Apply an additional effect preset to each frame

**Advanced Options:**

- ``--save-frames PATH`` - Directory to save individual frames
- ``--watermark TEXT`` - Add watermark text to frames
- ``--preview-filename PATH`` - Save a preview image
- ``--target-duration FLOAT`` - Stretch output to specified duration (seconds) using motion-compensated interpolation

**Quality Options:**

- ``--with-supersample`` - Apply 2x supersample anti-aliasing
- ``--with-fxaa`` - Apply FXAA anti-aliasing
- ``--with-ai`` - Apply AI image-to-image transformation
- ``--with-alt-text`` - Generate alt text description

**Examples:**

.. code-block:: bash

    # Create a basic animation
    noisemaker animate acid --frame-count 100 -o acid-loop.mp4

    # Create animation with effect applied
    noisemaker animate voronoi --effect-preset glitchin-out -o glitchy.mp4

    # Save individual frames
    noisemaker animate multires --save-frames ./frames/ -o anim.mp4

    # Create animation with specific duration
    noisemaker animate timeworms --target-duration 5.0 -o timed.mp4

apply
~~~~~

Apply an effect preset to an existing .png or .jpg image.

.. code-block:: bash

    noisemaker apply EFFECT_PRESET INPUT_FILENAME [OPTIONS]

**Required Arguments:**

- ``EFFECT_PRESET`` - Name of the effect preset to apply
- ``INPUT_FILENAME`` - Path to input image (.png or .jpg)

**Options:**

- ``--filename FILE`` - Output filename (default: mangled.png)
- ``--seed INTEGER`` - Random seed for stochastic effects
- ``--time FLOAT`` - Time value for animated effects
- ``--speed FLOAT`` - Animation speed modifier
- ``--no-resize`` - Don't resize image (may break some presets)
- ``--with-fxaa`` - Apply FXAA anti-aliasing

**Examples:**

.. code-block:: bash

    # Apply a glitch effect
    noisemaker apply glitchin-out photo.jpg -o glitched.jpg

    # Apply effect without resizing
    noisemaker apply vignette-dark image.png --no-resize -o output.png

    # Apply with anti-aliasing
    noisemaker apply bloom input.jpg --with-fxaa -o bloomed.jpg

    # Apply time-based effect
    noisemaker apply worms photo.png --time 0.5 -o warped.png

mashup
~~~~~~

Blend multiple images from a directory into a single composite.

.. code-block:: bash

    noisemaker mashup [OPTIONS]

**Required Options:**

- ``--input-dir DIRECTORY`` - Directory containing .png and/or .jpg images

**Optional Arguments:**

- ``--filename FILE`` - Output filename (default: mashup.png)
- ``--control-filename TEXT`` - Path to control image for blending
- ``--time FLOAT`` - Time value for animation
- ``--speed FLOAT`` - Animation speed
- ``--seed INTEGER`` - Random seed

**Examples:**

.. code-block:: bash

    # Blend all images in a directory
    noisemaker mashup --input-dir ./images/ -o combined.png

    # Blend with control image
    noisemaker mashup --input-dir ./photos/ --control-filename mask.png -o blend.png

magic-mashup
~~~~~~~~~~~~

Create an animated collage from multiple directories of image sequences.

.. code-block:: bash

    noisemaker magic-mashup [OPTIONS]

**Required Options:**

- ``--input-dir DIRECTORY`` - Directory containing subdirectories of frames

**Common Options:**

- ``--width INTEGER`` - Output width in pixels (default: 512)
- ``--height INTEGER`` - Output height in pixels (default: 512)
- ``--filename FILE`` - Output filename (default: mashup.mp4)
- ``--frame-count INTEGER`` - Number of frames to generate (default: 50)
- ``--seed INTEGER`` - Random seed
- ``--effect-preset NAME`` - Apply an effect preset to the collage

**Advanced Options:**

- ``--save-frames PATH`` - Directory to save individual frames
- ``--watermark TEXT`` - Add watermark text
- ``--preview-filename PATH`` - Save a preview image
- ``--target-duration FLOAT`` - Stretch output to specified duration (seconds)

**Examples:**

.. code-block:: bash

    # Create collage animation
    noisemaker magic-mashup --input-dir ./sequences/ -o collage.mp4

    # Create collage with effect
    noisemaker magic-mashup --input-dir ./frames/ --effect-preset vortex -o magic.mp4

Working with Presets
--------------------

Presets are predefined combinations of layers, effects, and settings. They are defined in ``dsl/presets.dsl``.

Common generator presets include:

- ``basic`` - Simple multi-octave noise
- ``multires`` - Multi-resolution noise
- ``voronoi`` - Voronoi cell patterns
- ``dla`` - Diffusion-limited aggregation
- ``fractal-smoke`` - Fractal smoke patterns
- ``acid`` - Psychedelic patterns
- ``timeworms`` - Animated worm-like patterns

Common effect presets include:

- ``bloom`` - Glow/bloom effect
- ``glitchin-out`` - Digital glitch artifacts
- ``vignette-dark`` - Dark vignette
- ``crt`` - CRT screen simulation
- ``posterize`` - Color posterization
- ``pixel-sort`` - Pixel sorting effect

Use ``random`` as the preset name to get a randomly selected preset.

Tips and Best Practices
------------------------

**Reproducibility:**

Use ``--seed`` to generate reproducible results:

.. code-block:: bash

    noisemaker generate acid --seed 42 -o output1.png
    noisemaker generate acid --seed 42 -o output2.png
    # output1.png and output2.png will be identical

**Performance:**

- Start with smaller dimensions (512x512) for testing
- Use ``--with-supersample`` or ``--with-fxaa`` for better quality at the cost of render time
- Higher ``--frame-count`` values will increase animation render time

**Output Formats:**

- Use ``.png`` for lossless output (larger files)
- Use ``.jpg`` for smaller files with some quality loss
- Animations are typically saved as ``.mp4`` or ``.gif``

**Debugging:**

Use ``--debug-print`` to see what settings a preset uses:

.. code-block:: bash

    noisemaker generate acid --debug-print -o test.png

API Keys
--------

Some features require API keys set as environment variables:

- **Stability AI** (for ``--with-ai`` and ``--with-upscale``): Set ``STABILITY_API_KEY``
- **OpenAI** (for ``--with-alt-text``): Set ``OPENAI_API_KEY``

.. code-block:: bash

    export STABILITY_API_KEY="your-key-here"
    export OPENAI_API_KEY="your-key-here"

Complete Command Reference
---------------------------

.. literalinclude:: noisemaker-help.txt
   :language: text
