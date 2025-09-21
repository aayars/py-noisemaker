# Noisemaker WebGPU Pipeline Specification

This document defines the WebGPU rendering pipeline used by the JavaScript
port of Noisemaker to evaluate presets entirely on the GPU.  It replaces the
current experimental GPU path with a deterministic, real-time pipeline that
mirrors the CPU execution model while avoiding CPU↔GPU buffer copies except for
the final presentation step.

## 1. Goals

* Preserve behavioural parity with the CPU preset pipeline: a preset expands
  into one generator stage followed by zero or more effect stages, with shared
  utility functions (convolution, blending, masks, palettes, etc.).
* Keep all intermediate surfaces resident on the GPU.  After uniforms are
  uploaded, the generation and effect chain executes without CPU interaction
  until the final output texture is resolved to the canvas swap chain (or a
  single read-back buffer when parity tests require CPU inspection).
* Achieve real-time rendering by reusing GPU resources, reducing pipeline
  recompilation, and performing work in compute passes sized for the active
  output resolution.
* Support hot parameter updates (e.g., UI sliders) without rebuilding shader
  modules or reallocating textures.
* Provide enough diagnostics and profiling hooks to validate parity against the
  CPU implementation and to tune performance.

Non-goals:

* Implementing new visual features that do not already exist on the CPU path.
* Supporting browsers without WebGPU or devices lacking compute shader support
  (the CPU path remains the fallback).

## 2. CPU Pipeline Recap

1. Evaluate the preset DSL to obtain a generator function and an ordered list of
   effect functions with bound arguments.
2. Invoke the generator to produce an RGBA floating-point surface in RAM.
3. Sequentially invoke each effect, reading the previous surface and writing a
   new one.  Utility helpers (blending, convolution, palette lookup, etc.) are
   called directly by generator/effect code.
4. Return the final surface to the caller for encoding, display, or further
   processing.

The GPU pipeline emulates this structure with GPU compute passes instead of CPU
function calls.

## 3. WebGPU Pipeline Overview

### 3.1 High-Level Flow

1. **Context acquisition** – Request a `GPUDevice` from the browser, configure a
   `GPUCanvasContext` for presentation, and build a reusable pool of bind-group
   layouts and pipeline layouts.
2. **Preset compilation** – Translate the preset (generator + effects) into a
   sequence of shader stage descriptors.  Each descriptor contains:
   * A WGSL source template identifier (for generator/effect/utility variants).
   * A list of uniforms, samplers, and storage bindings.
   * Static specialization constants (resolution, channel count, etc.).
3. **Resource allocation** – Create (or reuse) two RGBA32F storage textures for
   ping-pong rendering, uniform buffers per stage, and optional auxiliary
   textures (noise tables, LUTs, blue noise masks).
4. **Execution** – For each frame:
   * Update uniform buffers that changed since the previous frame.
   * Encode a command buffer that executes the generator compute pass followed
     by each effect pass, alternating the ping-pong textures.
   * Resolve the final texture to the swap-chain texture for presentation (or
     copy to a staging buffer for parity tests).
5. **Cleanup** – Destroy GPU resources when presets change or the canvas is
   resized; keep global caches alive across renders.

### 3.2 Stage Categories

* **Generator stage** – Produces the initial surface by writing into storage
  texture A.  No texture inputs are bound.
* **Unary effect** – Reads from the previous surface and writes to the other
  ping-pong texture.
* **Binary effect / blender** – Reads from the previous surface plus an
  auxiliary texture (e.g., mask, palette ramp) and writes to the other texture.
* **Reduction / global effects** – Implement via tiled compute kernels with
  shared memory (e.g., convolution) that read/write the ping-pong textures or
  temporary buffers.

## 4. Shader Architecture

### 4.1 WGSL Modules

* Each operation is backed by a WGSL template kept under `src/webgpu/shaders`.
  Templates expose entry points for generator and effect variants and import
  shared utility functions (noise evaluation, hashing, interpolation).
* The multires generator implementation resides in `src/webgpu/shaders/multires.wgsl` and is
  loaded via `src/webgpu/shaders.js` so browsers fetch raw WGSL rather than a giant JS string.
* Preset compilation performs lightweight string templating to inject constants
  (e.g., octave counts) and to inline effect-specific code paths, mirroring CPU
  “unrolling”.
* Utility functions (convolution kernels, gradient lookup, palette sampling)
  live in dedicated WGSL libraries compiled once and imported where needed.
* All shaders target workgroup size `(8, 8, 1)` by default; presets may override
  via specialization constants when a different tile size yields better
  occupancy.

### 4.2 Binding Model

Bind group layout (set = 0) shared across stages:

| Binding | Type                     | Usage                                               |
|---------|--------------------------|-----------------------------------------------------|
| 0       | Uniform buffer           | Stage parameters (floats, ints, bool flags).        |
| 1       | Uniform buffer (optional)| Time/seed/global constants shared by all stages.    |
| 2       | Storage texture (read)   | Previous surface (not bound for generator).         |
| 3       | Storage texture (write)  | Output surface.                                     |
| 4…      | Additional resources     | Aux textures/samplers required by the stage.        |

* Bindings 2 and 3 swap between ping-pong textures each stage.
* Additional bind groups provide immutable lookup textures (e.g., blue noise,
  permutation tables).  These are created once per device.

### 4.3 Uniform Packing

* Uniform buffers use std140-compatible struct packing to match CPU layout.
* Each stage defines a `StageUniforms` struct mirroring its CPU argument list.
* Scalars are 32-bit; booleans map to `u32` (0/1).  Vectors are multiples of
  four floats.
* A shared `FrameUniforms` struct supplies:
  * `vec2 resolution`
  * `f32 time`
  * `u32 seed`
  * `u32 frameIndex`

Uniform buffers are double-buffered to allow writing while the GPU reads the
previous contents.

## 5. Resource Management

### 5.1 Ping-Pong Surfaces

* Two RGBA32F storage textures sized to the active output resolution.
* Created with `GPUTextureUsage.STORAGE_BINDING | COPY_SRC | COPY_DST | TEXTURE_BINDING`.
* Recreated only when the resolution or format changes.

### 5.2 Auxiliary Buffers

* Noise tables, permute arrays, and gradient lookup textures are uploaded once
  at startup.  They stay in device-local memory.
* Convolution kernels and other small constant arrays live in uniform buffers.

### 5.3 Descriptor Caching

* Bind group layouts and pipeline layouts are cached by signature (list of
  bindings + shader variants).
* Compute pipelines are cached per stage template + specialization constants.
  Reuse pipelines across frames to avoid recompilation.

## 6. Execution Flow

### 6.1 Initialization

1. Request adapter with `powerPreference = "high-performance"` when possible.
2. Request device enabling required features (e.g., `float32-filterable` if the
   canvas presentation path requires filtering).
3. Configure the canvas context with format `rgba16float` (preferred) or
   `bgra8unorm` fallback.  Maintain a swap-chain-size matching the canvas.
4. Build global bind groups for immutable resources.

### 6.2 Preset Compilation

1. Traverse the preset AST (from the DSL evaluator) to produce a flattened list
   of `StageDescriptor` objects ordered as generator first, then effects.
2. For each descriptor:
   * Select the WGSL template identifier.
   * Compute specialization constants and uniform layout metadata.
   * Ensure required auxiliary resources are allocated.
   * Acquire or build the compute pipeline from cache.
3. Produce a `PresetProgram` containing the descriptors, cached pipelines, and
   uniform buffer views.

### 6.3 Frame Rendering

For each animation frame or manual render request:

1. Acquire the current swap-chain texture (unless rendering off-screen).
2. Update frame-uniform buffer with `time`, `frameIndex`, resolution, seed.
3. For each stage descriptor:
   * Update stage-uniform buffer if any parameters changed.
   * Acquire bind group with appropriate ping-pong textures bound.
   * Encode a compute pass dispatching over `(ceil(width/8), ceil(height/8), 1)`.
   * Swap the ping-pong indices.
4. If rendering to the screen:
   * Begin a render pass that draws a full-screen quad sampling the final
     storage texture via a `texture_2d<f32>` binding and writes to the swap-chain
     texture.
   * Otherwise, copy the final storage texture into a staging buffer for CPU
     parity checks.
5. Submit the command buffer and optionally resolve a fence or timestamp query
   for profiling.

### 6.4 CPU/GPU Synchronization

* Avoid awaiting `queue.onSubmittedWorkDone()` during steady-state rendering.
  Instead, rely on double-buffered uniforms and fences only when reading back
  data (tests) or when resizing resources.
* When parity tests require CPU inspection, copy the final texture into a
  `GPUBuffer` with `COPY_SRC` usage and map it once the commands finish.  This is
  the only permitted GPU→CPU transfer.

## 7. Utility Operations

### 7.1 Convolution

* Implement separable convolution kernels as two compute passes (horizontal and
  vertical) using shared memory to cache tiles.
* Kernel weights are provided via the stage-uniform buffer.

### 7.2 Blending and Masking

* Blending operations sample the previous surface and auxiliary textures (mask,
  palette) in the same compute shader.  The auxiliary textures are bound as
  read-only `texture_2d<f32>` with nearest filtering.

### 7.3 Randomness and Noise Tables

* Hash-based noise functions reuse the same permutation and gradient tables as
  the CPU path, uploaded as 1D/2D textures.
* Time-varying presets multiply the `frameIndex` and `time` uniforms into the
  hash to maintain parity.

## 8. Parameter Updates and Hot Reloading

* UI controls update the stage-uniform buffers via `queue.writeBuffer` or mapped
  ArrayBuffers.  Only dirty regions are written.
* Recompilation occurs only when the preset topology changes (generator/effect
  list), not when scalar parameters change.
* When presets change, existing textures are retained if the resolution matches;
  otherwise they are destroyed and recreated.

## 9. Integration with JavaScript Runtime

* The public API exposes `compilePreset(preset, device, resources)` returning a
  `PresetProgram` with an `execute(commandEncoder, frameUniforms)` method.
* The composer orchestrates CPU fallback vs GPU execution by feature detection.
* Profiling data (`webgpu` milliseconds) is measured using timestamp queries
  when supported; otherwise measure via `performance.now()` around
  `queue.submit`.
* Error handling surfaces WebGPU validation errors through `device.pushErrorScope`.

## 10. Testing and Parity Verification

* Unit tests compare GPU outputs against CPU references by running deterministic
  presets, copying the final texture into a CPU buffer, and asserting per-pixel
  differences within an epsilon.
* Integration tests render selected presets on a hidden canvas to ensure the
  entire chain (generator + effects) runs without CPU↔GPU copies until the final
  assertion.
* Automated tests run in browsers launched with WebGPU enabled flags as noted in
  `README-JS.md`.

## 11. Performance Considerations

* Batch command encoding for multiple presets when rendering thumbnail grids to
  amortize submission overhead.
* Prefer compute shaders for all stages; avoid fragment passes except for the
  final blit to the swap chain.
* Reuse mapped buffers and avoid `await` in hot paths to keep the render loop
  responsive.
* Support dynamic resolution scaling by resizing ping-pong textures while
  keeping pipelines intact.

## 12. Future Work

* Explore asynchronous pipeline compilation using `device.createComputePipelineAsync`.
* Investigate shared memory optimizations for high-cost effects (erosion,
  reverb) to reduce dispatch counts.
* Consider multi-surface presets (multiple outputs) by extending the stage
  descriptors with explicit target indices.
* Expose GPU timing data in the UI for live profiling.
