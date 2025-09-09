# Noisemaker Vanilla JavaScript Port Specification

## 1. Goals and Scope

* Reproduce the reusable library portion of the Python project in plain JavaScript.
* Runtime target: WebGL2 in modern browsers.
* Node.js may be used for bundling, unit tests and regression scripts only; the final library must not depend on Node APIs.
* Support deterministic results through seed control and provide feature parity with the Python implementation where feasible.
* Out of scope: existing CLI tooling, rendering scripts, Stable Diffusion integration and any non-library utilities.

## 2. High‑level Architecture

### 2.1 Module Layout
The library is organised as ES modules inside `src/`.

| Module | Responsibility |
|---|---|
| `constants.js` | Enumerations and static lookup tables. |
| `simplex.js` | 3‑D OpenSimplex noise and loopable random helpers. |
| `value.js` | Core tensor math: value noise generation, resampling, convolution, derivatives, blending utilities. |
| `points.js` | Procedural point‑cloud generators used by Voronoi and DLA algorithms. |
| `masks.js` | Static bitmap/procedural masks and glyph atlas helpers. |
| `effectsRegistry.js` | Decorator‑like helper that stores effect metadata and callbacks. |
| `effects.js` | Library of post‑processing effects built on top of `value.js` utilities. |
| `composer.js` | Preset interpreter and pipeline orchestrator. |
| `palettes.js` | Cosine palette definitions. |
| `presets.js` | Declarative preset graph describing layered effect pipelines. |
| `util.js` | Miscellaneous utilities: tensor helpers, canvas export, logging. |
| `oklab.js` | RGB⇄OKLab colour‑space conversion routines. |
| `glyphs.js` | Font discovery and glyph atlas rasterisation using `CanvasRenderingContext2D`. |

### 2.2 Data Representation

* **Tensor** – wrapper around a WebGL texture representing a 3‑D float array `[height, width, channels]`.
* `Tensor.fromArray(Float32Array, shape)` uploads data to a floating‑point texture.
* `Tensor.read()` downloads pixels to a typed array for tests or export.
* Fallback CPU paths mirror the API using typed arrays when WebGL2 or float textures are unavailable.

### 2.3 Rendering Pipeline

1. Create a WebGL2 context bound to a `<canvas>`.
2. Each operation is a shader program that reads one or more input textures and writes to an FBO‑backed texture.
3. Operations are chained by ping‑ponging between textures; the final texture is presented on the canvas or converted to image data.

## 3. Module Specifications

### 3.1 `constants.js`
* Implement enumerations as `Object.freeze`d maps.
* Provide: `DistanceMetric`, `InterpolationType`, `PointDistribution`, `ValueDistribution`, `ValueMask`, `VoronoiDiagramType`, `ColorSpace`, `WormBehavior`, etc.
* Include helper predicates (e.g. `isAbsolute(metric)`, `isCenterDistribution(distrib)`).

### 3.2 `simplex.js`
* Port the 4‑D OpenSimplex algorithm to produce loopable noise.
* `random(time, seed, speed)` – periodic scalar valued helper.
* `simplex(shape, {time, seed, speed})` – returns a `Tensor` of `[height,width,(channels)]` with values in `[0,1]`.
* Expose `setSeed()` and `getSeed()` for deterministic sequences.

### 3.3 `value.js`
* Implements the fundamental math used by all higher level features.
* Functions operate on `Tensor` instances.
* Key routines:
  * `values(freq, shape, {distrib, corners, mask, maskInverse, maskStatic, splineOrder, time, speed})` – base value‑noise generator supporting all `ValueDistribution` types and mask application.
  * Resampling: `downsample`, `upsample`, `fft`, `ifft`, `warp`, `refract`, `convolution`, `ridge`, `rotate`, `zoom`.
  * Blending utilities: `blend(a,b,t)`, `normalize`, `clamp01`, `distance(dx,dy,metric)`.
  * Derivatives and filtering: `sobel`, `fxaa`, `gaussianBlur`.
  * Palette mapping: `valueMap`, `hsvToRgb`, `rgbToHsv`.
* All operations are expressed as GLSL snippets composed into shader programs.

### 3.4 `points.js`
* Generates two parallel arrays `(x[], y[])` in pixel coordinates for a given `freq` and distribution.
* Supports grid, waffle, chess, hexagonal, spiral, circular, concentric, rotating, and ValueMask‑driven point sets.
* Includes drift and multi‑generation growth used for DLA/Voronoi.

### 3.5 `masks.js`
* Hard‑coded bitmap masks (`ValueMask` enums) stored as nested arrays.
* Procedural masks such as Truchet tiles or invader patterns computed on demand.
* `maskValues(type, shape, {atlas, inverse, static})` returns a `Tensor` and an optional atlas texture.
* Glyph atlas loader reads user fonts, rasterises characters to textures and caches results.

### 3.6 `effectsRegistry.js`
* `register(name, fn, defaults)` records effect metadata.
* `EFFECTS` map is exported for lookups.
* `EFFECT_METADATA` exposes default parameters for each effect without the callback.
* Validation ensures every effect accepts `(tensor, shape, time, speed, ...params)`.

### 3.7 `effects.js`
* Over 70 shader‑driven effects; grouped for implementation clarity.
  * **Sampling/geometry:** `warp`, `reindex`, `funhouse`, `kaleidoscope`, `repeat`, `rotate`, `zoom`, `reflect`.
  * **Lighting/shading:** `shadow`, `bloom`, `vignette`, `brcosa`, `levels`.
  * **Stylisation:** `posterize`, `dither`, `halftone`, `valueMask`, `glyphMap`.
  * **Noise based:** `fbm`, `worms`, `grain`, `erosionWorms`.
  * **Post colour:** `palette`, `saturation`, `invert`, `aberration`, `randomHue`.
* Every effect is defined as a JS function that builds or reuses shader programs and returns a new `Tensor`.
* All effects are registered through `effectsRegistry.register` for discovery by the composer.

### 3.8 `composer.js`
* `Preset` class holds resolved settings and ordered effect lists.
* `render(presetName, seed, {width,height,colorSpace,withAlpha,time,speed})` constructs the generator pipeline, applies octave‑level and post effects, and presents the result to the canvas.
* Supports inheritance between presets, unique layers, and inline overrides.
* Provides convenience wrappers `Effect(name, params)` for programmatic composition.

### 3.9 `palettes.js`
* Export `PALETTES` object describing cosine palettes (`amp`, `freq`, `offset`, `phase`).
* Utility `samplePalette(name, t)` returns RGB triplets.

### 3.10 `presets.js`
* Contains a lazy function `PRESETS()` returning an object of preset definitions.
* Presets specify `layers`, `settings`, optional `post`, `final`, and `ai` placeholders (ignored in JS port).
* Each metadata function returns data based on randomised helpers and enumerations.

### 3.11 `util.js`
* Canvas export: `savePNG(tensor, filename)` and `tensorFromImage(image)` using `<canvas>` APIs.
* Logger abstraction, seeded random helpers, and shape utilities.
* Colour helpers: `fromSRGB`, `toSRGB`, `linToSRGB`, `srgbToLin`.

### 3.12 `oklab.js`
* Functions `rgbToOklab(tensor)` and `oklabToRgb(tensor)` implemented in pure JS math on `Tensor` data.

### 3.13 `glyphs.js`
* `loadFonts()` reads `.ttf` files from a configurable directory when running under Node.
* `loadGlyphs(shape)` rasterises printable ASCII glyphs to monochrome textures sorted by brightness for use in value masks and glyph map effects.

## 4. Testing and Build

* Unit tests are written with Node’s built‑in `assert` module and run via `npm test`.
* Numerical tests compare selected operations against reference outputs produced by the Python version.
* Web tests use headless `chrome --use-gl=swiftshader` to exercise shader paths.
* Build step produces both ES module and UMD bundles using a minimal script (no bundler dependency).

## 5. Implementation Roadmap

1. **Core infrastructure:** WebGL2 context manager, `Tensor` wrapper, shader compilation utilities.
2. **Noise foundations:** implement `constants`, `simplex`, `value` (basic distributions, blending, distance).
3. **Point/mask utilities:** `points`, `masks`, glyph atlas.
4. **Effect framework:** `effectsRegistry` and initial subset of effects for parity tests.
5. **Composer and presets:** port palette handling, preset resolution, rendering pipeline.
6. **Full effect library:** iteratively port remaining effects and colour utilities.
7. **Testing harness and build scripts.**

## 6. Non‑Goals & Future Work

* No command‑line interface or Stable Diffusion integrations.
* No automatic documentation generation; the spec itself is authoritative during the port.
* Future enhancements may leverage WebGPU once widely available, but initial target remains WebGL2.
