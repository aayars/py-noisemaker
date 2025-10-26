This directory (/shaders) contains the Noisemaker shader collection, where we are attempting to create faithful shader reproductions of Noisemaker's effects.

- *MANDATORY*: If you modify a shader, it must load and render in /demo/gpu-effects/index.html without console errors.

- *OBVIOUS*: Visual diffs must be performed with the same *seed*, *time*, and *frame*

- *IMPORTANT*: Do not produce documentation unless requested.

- Favor re-use of tests/visual-diff-effect.js over littering the repo with garbage files.

- Use the development server on port 8080. Do not start new servers.

- The effects selector element ID on the gpu-effects demo is "effect-selector".

- **READ THE IMPLEMENTATION GUIDE**: Before implementing any effect, read `/shaders/IMPLEMENTATION_GUIDE.md` for complete architecture documentation, tutorials, and working examples.

- This shader collection is completely separate from Noisemaker's JS and Python pipelines. Noisemaker's Python implementation, under /noisemaker, is our reference platform for implementing similar visual outputs.

- Shaders are under /shaders/effects.

- Do not touch /noisemaker or /js/noisemaker if you were sent here to work on shaders.

- Our primary shader viewer is at /demo/gpu-effects/index.html (WebGPU). Keep controls and uniforms consistent with shader implementations.

- Keep shaders consistent with each other and with Noisemaker algorithms.

- Shader controls should match effect params in the Python reference, exactly or as closely as possible. Ignore "shape", which we are handling differently in shader land.

- Follow best authoring practices for WGSL shaders.

- Shader tests are under /shaders/tests and /test (Playwright tests for WebGPU demo).

- Important: Our shaders and textures are in a 4-channel RGBA format. Don't waste cycles converting, except in cases where *explicit* color space conversion is called for. This means: Channel count is always 4. Do not waste effort counting channels, the number will always be 4. Repeat that out loud.

- Important: You must end struct member declarations with ",", not ";". If you use a semicolon, the file can not be parsed as wgsl. Repeat that out loud.

---

# Effect Implementation Quick Reference

## For Simple Single-Pass Effects
1. Extend `SimpleComputeEffect` (see `/shaders/effects/vignette/effect.js`)
2. Use explicit offset parameter bindings in `meta.json`
3. Follow the single-pass tutorial in IMPLEMENTATION_GUIDE.md

## For Complex Multi-Pass Effects
1. Implement custom effect class WITHOUT extending SimpleComputeEffect (see `/shaders/effects/worms/effect.js`)
2. Define `resources.computePasses` array to drive multi-pass execution
3. Use `beforeDispatch()` and `afterDispatch()` hooks for buffer swapping
4. Follow the multi-pass tutorial in IMPLEMENTATION_GUIDE.md

## Key Architecture Points
- **Runtime never calls `execute()`**—passes are driven by `resources.computePasses` array in `main.js`
- **Two binding formats**: explicit offsets (for SimpleComputeEffect) or dot-notation (for custom classes)
- **Uniform buffers**: Size must be multiple of 4 bytes; runtime upscales to minimum 256 bytes
- **Multi-pass**: Each pass specifies `pipeline`, `bindGroup`, `workgroupSize`, and `getDispatch` function

---

# ✧ The Sacred Commandments of WGSL ✧

## I. On Syntax and Purity

1. **Thou shalt obey the spec.**
   WGSL is strictly defined; deviations, dialects, or experimental extensions are forbidden.
2. **Thou shalt use `@group` and `@binding` explicitly.**
   Never rely on implicit bindings. Clarity is law.
3. **Thou shalt always specify storage classes.**
   Use `var<uniform>`, `var<storage, read_write>`, etc. Explicitness prevents sin.

## II. On Naming and Structure

4. **Thou shalt name uniforms, varyings, and workgroup variables with care.**
   Descriptive and consistent names guide both compiler and mortal.
5. **Thou shalt group related bindings in structs.**
   Flatten not the binding table; cohesion brings order.
6. **Thou shalt mark all entry points with `@vertex`, `@fragment`, or `@compute`.**
   There is no entry without a sigil.

## III. On Types and Safety

7. **Thou shalt use explicit types everywhere.**
   `let x: f32 = 0.0` is holier than `let x = 0.0`.
8. **Thou shalt avoid implicit widening or narrowing.**
   Cast between `u32`, `i32`, and `f32` without sloth.
9. **Thou shalt prefer `vecN` and `matN` constructors over ad-hoc component filling.**

## IV. On Layout and Alignment

10. **Thou shalt align structs according to WGSL’s holy layout rules.**
    Pad with `@size(N)` where necessary to appease the GPU.
11. **Thou shalt match host-side structs exactly.**
    Let no field be misaligned, lest debugging become torment.

## V. On Control and Logic

12. **Thou shalt prefer `switch` over cascaded `if` where semantically clean.**
13. **Thou shalt avoid unbounded loops.**
    Use `for` or `loop` constructs with explicit exits; the validator demands it.
14. **Thou shalt use `select()` for branchless expressions where readability permits.**

## VI. On Precision and Performance

15. **Thou shalt not abuse `f16` unless measured.**
    Half precision is a gift, but only in wise hands.
16. **Thou shalt minimize texture sampling inside loops.**
17. **Thou shalt move uniform and constant expressions out of the hot path.**

## VII. On Entry Point Contracts

18. **Thou shalt explicitly decorate all I/O.**
    Use `@location`, `@builtin`, `@interpolate` where mandated.
19. **Thou shalt never assume coordinate conventions.**
    Always recall: WGSL fragment `@builtin(position)` is upper-left origin.
20. **Thou shalt ensure entry point outputs are fully defined.**
    No output may be left unwritten.

## VIII. On Modularity

21. **Thou shalt divide code into reusable `fn` functions.**
    Entry points are temples, not dumping grounds.
22. **Thou shalt not duplicate constants.**
    Use `const` and share across modules.
23. **Thou shalt avoid magic numbers.**
    Every constant deserves a name.

## IX. On Error and Validation

24. **Thou shalt heed the validator.**
    Its warnings are prophecy; ignore them not.
25. **Thou shalt test shaders across multiple backends (Dawn, wgpu, WebKit).**
    What runs on one GPU may fail on another.

## X. Eternal Law

26. **Thou shalt write for determinism.**
    GPU non-determinism is chaos; order your math.
27. **Thou shalt document all assumptions.**
    Sacred comments explain binding indices, coordinate spaces, and conventions.
28. **Thou shalt not import host logic into shader logic.**
    Each side of the API has its domain; respect the boundary.

## WGSL Style Guide

The existing shaders already follow broadly consistent conventions—mirror them when
adding or revising code so the collection reads like a single codebase.

### Formatting

- **Indentation:** Use 4 spaces for each indentation level. This applies to block
  bodies, struct members, control flow, and long argument lists. Tabs are forbidden.
- **Line width:** Aim to keep lines ≤ 100 characters. Break long expressions across
  multiple lines with subsequent lines indented one additional level (4 more spaces).
- **Spacing:** Use spaces around `:` in type annotations (`let value : f32`), around
  binary operators, and after commas. Keep attribute lists tight: `@group(0)
  @binding(0)`.
- **Braces:** Opening braces sit on the same line as declarations. Always include
  closing braces aligned with the start of the construct.

### Naming

- **Types and structs:** Use `PascalCase`, e.g. `struct TintParams`.
- **Functions:** Use `lower_snake_case`, with the exception of entry points, which
  remain `fn main` (or `main_fragment`/`main_vertex` when multiple entry points are
  needed).
- **Constants:** Prefer `SCREAMING_SNAKE_CASE`, e.g. `const CHANNEL_COUNT : u32 = 4u;`.
- **Bindings and locals:** Use descriptive `lower_snake_case` identifiers such as
  `input_texture`, `output_buffer`, `workgroup_index`.
- **Private helper modules:** When a shader grows large, group helpers in logical
  clusters separated by `// ----------` comment blocks so navigation stays easy.

### Standard Fixtures

- Declare the canonical bindings up front and in this order when present:
  1. `@group(0) @binding(0) var input_texture : texture_2d<f32>;`
  2. `@group(0) @binding(1) var<storage, read_write> output_buffer : array<f32>;`
  3. `@group(0) @binding(2) var<uniform> ParamsStruct;`
  Additional bindings (samplers, lookup textures, storage buffers) follow afterwards
  with monotonically increasing binding indices.
- Uniform structs are named `<EffectName>Params` and pack scalars into `vec4<f32>`
  groups to respect WGSL alignment. Pad unused slots explicitly (e.g. `_pad0`).
- Define shared constants (like `CHANNEL_COUNT`) once near the top of the file.
- Default compute entry points use `@compute @workgroup_size(8, 8, 1)` unless the
  algorithm requires a different tile size. Document any deviations with a comment.
- Emit a short header comment explaining the effect, mirroring the Python
  implementation description and noting any key assumptions or differences.

### Control Flow & Helpers

- Guard out-of-bounds workgroup invocations early (`if (gid.x >= width_u || ...)
  return;`).
- Extract reusable math into helper functions (e.g. `clamp_coord`, `wrap_index`)
  rather than inlining identical expressions across shaders.
- When mapping CPU utilities (value noise, ridge transforms, etc.), give helpers the
  same name as the Python source for easy cross-reference.

---
