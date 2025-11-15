WebGPU API
==========

Noisemaker includes an experimental collection of WebGPU compute shader effects that run natively in the browser. These shaders provide GPU-accelerated image processing independent of the Python and JavaScript implementations.

.. warning::
   **Experimental & Under Construction**: The shader effects are heavily under development with unoptimized and untested code. Use with caution and expect breaking changes.

Overview
--------

The shader collection represents a parallel implementation of Noisemaker effects using WGSL (WebGPU Shading Language) compute shaders. Each effect runs entirely on the GPU, processing images in real-time within the browser.

Key Characteristics
~~~~~~~~~~~~~~~~~~~

* **GPU-native** - Effects run as WebGPU compute shaders
* **Browser-based** - No server or build step required
* **Real-time rendering** - Interactive parameter adjustment
* **Independent** - Separate from Python/JavaScript implementations
* **Experimental** - Active development with incomplete features

Architecture
------------

Effect Structure
~~~~~~~~~~~~~~~~

Each shader effect lives in ``/shaders/effects/<effect-name>/``:

.. code-block:: text

    shaders/effects/<effect-name>/
    ├── effect.js           # JavaScript effect class
    ├── meta.json          # Metadata: parameters, bindings
    └── <effect-name>.wgsl # WGSL compute shader

Effect Types
~~~~~~~~~~~~

**Simple Effects** (extend ``SimpleComputeEffect``):

- Single-pass compute operations
- Automatic parameter management
- Standard binding system
- Examples: vignette, posterize, blur, sobel

**Complex Effects** (custom implementation):

- Multi-pass rendering
- Custom resource management
- Manual buffer handling
- Examples: worms, wormhole, dla, fibers

WebGPU Pipeline
~~~~~~~~~~~~~~~

1. **Input Texture** - Source image loaded into GPU texture
2. **Uniform Buffer** - Effect parameters (width, height, custom params)
3. **Compute Shader** - WGSL shader processes pixels
4. **Storage Buffer** - Output data (typically RGBA float32)
5. **Presentation** - Results displayed on canvas

Runtime API
-----------

The shader demo and tests share a small JavaScript runtime under ``shaders/src``. These
modules provide reusable WebGPU wiring, effect lifecycle helpers, and UI generation.

``createWebGPURuntime(config)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defined in ``shaders/src/runtime/webgpu-runtime.js``. Creates a singleton-style runtime
with cache management and WebGPU device access. ``config`` accepts:

* ``canvas`` *(required)* – Target ``HTMLCanvasElement``.
* ``getShaderDescriptor`` *(required)* – Function that returns `{ id, url, label, entryPoint, resources }` metadata for a shader.
* ``parseShaderMetadata`` *(required)* – Function that parses WGSL source into structured metadata (see ``shader-registry``).
* Logging hooks: ``logInfo``, ``logWarn``, ``logError``.
* ``setStatus`` – Status updater for UI.
* ``fatal`` – Error handler that throws or aborts.
* ``onCachesCleared`` – Optional callback after GPU caches are flushed.
* ``onDeviceInvalidated`` – Optional callback when the device becomes unusable.

The return value exposes:

* ``ensureWebGPU()`` – Lazily request adapter/device, reusing across calls.
* ``getWebGPUState({ alphaMode })`` – Configure the canvas context and return ``{ adapter, device, webgpuContext }``.
* ``clearGPUObjectCaches()`` – Drop cached shader modules, pipelines, and bind group layouts.
* ``createShaderResourceSet(...)`` – Allocate buffers/textures based on effect metadata.
* ``getOrCreateComputePipeline(...)`` – Compile and cache compute pipelines per shader/entry point.
* ``renderFlatColor({ alphaMode, clearColor })`` – Simple render pass useful for diagnostics.
* Pipeline caches: ``computePipelineCache``, ``pipelineCache``, ``blitShaderModuleCache`` (maps are returned for advanced use).

``EffectManager``
~~~~~~~~~~~~~~~~~

Located at ``shaders/src/runtime/effect-manager.js``. Manages effect registration, lazy
loading, parameter updates, and cleanup. Key methods:

* ``registerEffect({ id, label, metadata, loadModule })`` – Register effects with dynamic ``import`` loaders.
* ``setActiveEffect(id)`` – Switch to a different effect, instantiating on demand.
* ``updateActiveParams(updates)`` – Apply parameter changes; delegates to the active effect.
* ``getActiveUIState()`` – Return current slider/toggle state for UI rendering.
* ``invalidateActiveEffectResources()`` – Ask the active effect to drop GPU resources when the device resets.

Instantiate the manager with shared helpers: ``new EffectManager({ helpers })``. The demo
passes logging utilities and runtime helpers (pipeline builders, resource factories) so
effects can share caches.

``SimpleComputeEffect``
~~~~~~~~~~~~~~~~~~~~~~~

The base class for most effects lives in ``shaders/common/simple-compute-effect.js``. A
subclass only needs to:

1. Declare static ``metadata`` (imported from ``meta.json``).
2. Optionally override ``getResourceCreationOptions`` or lifecycle hooks such as
     ``onResourcesCreated``.

The base class provides:

* Automatic parameter coercion and GPU buffer writes based on ``parameterBindings``.
* ``ensureResources({ device, width, height, multiresResources })`` – Allocates GPU
    resources and caches them per size/device.
* ``updateParams(updates)`` – Handles booleans vs. numeric types with tolerance checks.
* ``invalidateResources()`` – Disposes buffers/textures safely.

For complex multi-pass effects, authors can skip ``SimpleComputeEffect`` and implement a
custom class that consumes the same helpers provided by ``EffectManager`` and the runtime.

Browser Demo
------------

Interactive Viewer
~~~~~~~~~~~~~~~~~~

The primary shader viewer is located at ``/demo/gpu-effects/index.html``:

.. code-block:: bash

    # Serve locally (from project root)
    python3 -m http.server 8080
    
    # Open in browser
    open http://localhost:8080/demo/gpu-effects/

Features
~~~~~~~~

* **Effect selector** - Browse all available shader effects
* **Parameter controls** - Adjust effect parameters in real-time
* **Animation** - Time-based effects with speed control
* **Canvas modes** - Fixed or full-bleed rendering
* **Seed control** - Deterministic random generation
* **Frame counter** - Track animation progress

Browser Requirements
~~~~~~~~~~~~~~~~~~~~~

* Chrome 113+ or Edge 113+ (WebGPU support)
* Hardware with GPU compute shader support
* Sufficient VRAM for texture operations

Available Effects
-----------------

Color Adjustments
~~~~~~~~~~~~~~~~~

* **adjust_brightness** - Modify image brightness
* **adjust_contrast** - Adjust contrast levels
* **adjust_hue** - Shift color hue
* **adjust_saturation** - Control color saturation
* **tint** - Apply color tinting
* **color_map** - Remap colors via lookup

Blur & Distortion
~~~~~~~~~~~~~~~~~

* **blur** - Gaussian blur
* **lens_distortion** - Barrel/pincushion distortion
* **lens_warp** - Lens-based warping
* **wobble** - Wave-based distortion
* **ripple** - Ripple effect
* **vortex** - Swirl/vortex distortion
* **warp** - Arbitrary image warping

Stylization
~~~~~~~~~~~

* **posterize** - Reduce color levels
* **pixel_sort** - Sort pixels by criteria
* **lowpoly** - Low-polygon aesthetic
* **sketch** - Pencil sketch effect
* **grime** - Dirt/grime overlay
* **scratches** - Surface scratches
* **vaseline** - Soft focus effect

Edge Detection
~~~~~~~~~~~~~~

* **sobel** - Sobel edge detection
* **outline** - Edge outlining
* **glowing_edges** - Luminous edge effect
* **derivative** - Image derivatives
* **normal_map** - Generate normal maps

Noise & Grain
~~~~~~~~~~~~~

* **grain** - Film grain
* **snow** - Static/snow noise
* **spatter** - Splatter patterns
* **nebula** - Nebula-like clouds
* **clouds** - Cloud generation

Lighting & Shading
~~~~~~~~~~~~~~~~~~

* **bloom** - Glow/bloom effect
* **shadow** - Shadow rendering
* **light_leak** - Lens light leaks
* **vignette** - Vignette darkening

Retro Effects
~~~~~~~~~~~~~

* **crt** - CRT monitor simulation
* **vhs** - VHS tape artifacts
* **scanline_error** - Scanline glitches
* **degauss** - Degaussing effect
* **jpeg_decimate** - JPEG compression artifacts

Procedural
~~~~~~~~~~

* **worms** - Meandering worm patterns
* **erosion_worms** - Erosion simulation
* **dla** - Diffusion-limited aggregation
* **fibers** - Fiber/hair generation
* **voronoi** - Voronoi diagrams
* **stray_hair** - Hair strand overlay

Geometric
~~~~~~~~~

* **kaleido** - Kaleidoscope effect
* **rotate** - Image rotation
* **reindex** - Pixel reordering
* **frame** - Add frames/borders
* **simple_frame** - Basic framing

Convolution
~~~~~~~~~~~

* **convolve** - Custom convolution kernels
* **conv_feedback** - Feedback convolution
* **fxaa** - Fast approximate anti-aliasing
* **ridge** - Ridge enhancement

Special
~~~~~~~

* **aberration** - Chromatic aberration
* **false_color** - False color mapping
* **density_map** - Density visualization
* **reverb** - Echo/reverb effect
* **sine** - Sinusoidal transforms
* **smoothstep** - Smooth interpolation
* **value_refract** - Value-based refraction
* **refract** - Refraction simulation
* **wormhole** - Wormhole/tunnel effect
* **glyph_map** - Text/glyph rendering
* **palette** - Palette application
* **texture** - Texture overlay
* **spooky_ticker** - Animated text ticker
* **on_screen_display** - OSD overlay

Implementation Guide
--------------------

For developers implementing new shader effects, see ``/shaders/IMPLEMENTATION_GUIDE.md`` for complete documentation including:

* Effect lifecycle and architecture
* Parameter binding system (explicit offsets vs. dot-notation)
* Single-pass effect tutorial
* Multi-pass effect tutorial
* WGSL shader structure requirements
* Testing guidelines
* Common pitfalls and solutions

Simple Effect Example
~~~~~~~~~~~~~~~~~~~~~

**meta.json**:

.. code-block:: json

    {
      "parameters": {
        "amount": {
          "type": "float",
          "default": 1.0,
          "min": 0.0,
          "max": 2.0,
          "description": "Effect intensity"
        }
      },
      "parameterBindings": {
        "width": { "buffer": "params", "offset": 0 },
        "height": { "buffer": "params", "offset": 1 },
        "channel_count": { "buffer": "params", "offset": 2 },
        "amount": { "buffer": "params", "offset": 3 }
      }
    }

**effect.wgsl**:

.. code-block:: wgsl

    struct Params {
        width: f32,
        height: f32,
        channel_count: f32,
        amount: f32,
    }
    
    @group(0) @binding(0) var input_texture : texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
    @group(0) @binding(2) var<uniform> params : Params;
    
    @compute @workgroup_size(8, 8, 1)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let width_u = u32(params.width);
        let height_u = u32(params.height);
        
        if (gid.x >= width_u || gid.y >= height_u) {
            return;
        }
        
        let pixel = textureLoad(input_texture, vec2<i32>(gid.xy), 0);
        let base_index = (gid.y * width_u + gid.x) * 4u;
        
        // Apply effect
        output_buffer[base_index + 0u] = pixel.r * params.amount;
        output_buffer[base_index + 1u] = pixel.g * params.amount;
        output_buffer[base_index + 2u] = pixel.b * params.amount;
        output_buffer[base_index + 3u] = pixel.a;
    }

**effect.js**:

.. code-block:: javascript

    import { SimpleComputeEffect } from '../../common/SimpleComputeEffect.js';
    
    export class MyEffect extends SimpleComputeEffect {
        static metadata = {
            effectName: 'my_effect',
            effectLabel: 'My Effect',
            metadataPath: '/shaders/effects/my_effect/meta.json',
            shaderPath: '/shaders/effects/my_effect/my_effect.wgsl'
        };
    }

Testing
-------

Shader Test Suite
~~~~~~~~~~~~~~~~~

Tests are located in ``/shaders/tests/`` and use Puppeteer for headless browser testing:

.. code-block:: bash

    cd shaders/tests
    npm install
    npm test

Visual Regression
~~~~~~~~~~~~~~~~~

Visual diff testing compares shader output against Python reference:

.. code-block:: bash

    node shaders/tests/visual-diff-effect.js <effect-name>

**Important**: Visual comparisons must use identical seed, time, and frame values to be valid.

Common Issues
~~~~~~~~~~~~~

* **Browser hangs** - Effect may have infinite loops or excessive computation
* **Black output** - Check binding indices and buffer sizes
* **Incorrect colors** - Verify channel order (RGBA) and normalization
* **Memory errors** - Ensure buffer sizes are multiples of 4 bytes

WGSL Style Guide
----------------

Critical Rules
~~~~~~~~~~~~~~

1. **Struct members** end with ``,`` not ``;``
2. **All textures are 4-channel RGBA** - don't count channels dynamically
3. **Explicit bindings** - Always use ``@group(0) @binding(N)``
4. **Match offsets exactly** - JavaScript offset must match WGSL struct layout
5. **Guard bounds** - Check ``gid.x >= width_u || gid.y >= height_u`` early

Naming Conventions
~~~~~~~~~~~~~~~~~~

* **Types/structs**: ``PascalCase`` (e.g., ``MyEffectParams``)
* **Functions**: ``snake_case`` (e.g., ``apply_effect``)
* **Constants**: ``SCREAMING_SNAKE_CASE``
* **Bindings**: ``snake_case`` (``input_texture``, ``output_buffer``, ``params``)

Standard Bindings
~~~~~~~~~~~~~~~~~

Most effects follow this pattern:

.. code-block:: wgsl

    @group(0) @binding(0) var input_texture : texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;
    @group(0) @binding(2) var<uniform> params : ParamsStruct;

Alignment Requirements
~~~~~~~~~~~~~~~~~~~~~~

Uniform buffer structs must respect WGSL alignment:

* ``f32`` - 4-byte aligned
* ``vec2<f32>`` - 8-byte aligned
* ``vec3<f32>`` - 16-byte aligned (padded!)
* ``vec4<f32>`` - 16-byte aligned

Pack scalars into ``vec4`` groups to minimize padding.

Development Workflow
--------------------

Creating New Effects
~~~~~~~~~~~~~~~~~~~~

1. Create directory: ``/shaders/effects/my_effect/``
2. Add metadata: ``meta.json`` with parameters and bindings
3. Write shader: ``my_effect.wgsl`` with compute entry point
4. Create class: ``effect.js`` extending ``SimpleComputeEffect`` or custom
5. Add to manifest: Update ``/shaders/manifest.json``
6. Test in demo: Load effect in GPU effects viewer
7. Debug: Check browser console for WebGPU errors

Hot Reloading
~~~~~~~~~~~~~

The demo viewer supports hot reloading:

* Shader changes require page reload
* Parameter changes apply immediately
* Metadata changes require effect reselection

Performance Tips
~~~~~~~~~~~~~~~~

* Minimize texture reads - cache values in registers
* Avoid branching in inner loops
* Use workgroup shared memory for collaboration
* Batch operations to reduce dispatches
* Profile with browser DevTools GPU timings

Known Limitations
-----------------

Current State
~~~~~~~~~~~~~

* **Incomplete** - Many effects are unfinished or non-functional
* **Unoptimized** - Performance has not been tuned
* **Untested** - Limited test coverage
* **Breaking changes** - API may change without notice
* **Browser-specific** - Only tested in Chrome/Edge

Missing Features
~~~~~~~~~~~~~~~~

* Comprehensive parity testing against Python
* Shader compilation error reporting in UI
* Performance profiling tools
* Multi-effect chaining
* Preset system integration

Stability Warnings
~~~~~~~~~~~~~~~~~~

Effects known to cause issues:

* Large DLA generations may hang browser
* Multi-pass effects with feedback can overflow memory
* Complex procedural effects may timeout GPU

Further Reading
---------------

* `Shader Implementation Guide </shaders/IMPLEMENTATION_GUIDE.md>`_
* `Shader Agent Instructions </shaders/AGENTS.md>`_
* `WebGPU Specification <https://www.w3.org/TR/webgpu/>`_
* `WGSL Language Spec <https://www.w3.org/TR/WGSL/>`_
* `GPU Effects Demo </demo/gpu-effects/index.html>`_
