# WGSL Multires Shader TODO

The WGSL module now exposes a richer uniform struct (`StageUniforms`) and
implements multi-octave accumulation for simplex noise, including falloff and
reduce-max blending paths. Alpha blending currently uses a provisional
pre-multiplied mix while we bootstrap mask aware layers. The shader now
<<<<<<< ours
evaluates per-channel simplex seeds, performs HSV/OKLab conversion, applies the
HSV hue/saturation/brightness overrides (including brightness frequency
overrides), and honours reduce-max/alpha octave blending, but it is still not
wired into the runtime. Follow-up work should address the remaining gaps before
enabling the stage:
=======
evaluates per-channel simplex seeds, performs HSV/OKLab conversion, and applies
basic hue/saturation/sine adjustments. The stage is wired into the WebGPU
pipeline and uniforms are populated through the custom resolver in
`pipeline.js`, but a number of fidelity gaps remain before we can call the
generator production ready:
>>>>>>> theirs

The shader lives at `src/webgpu/shaders/multires.wgsl` and is loaded on demand
via `src/webgpu/shaders.js`, so future updates should modify that WGSL module
directly.

<<<<<<< ours
<<<<<<< ours
1. **Uniform layout integration** – update the WebGPU pipeline to populate the
<<<<<<< ours
   expanded `StageUniforms` fields (`brightness_freq`, `options2`, `options3`,
   etc.) and document the packing strategy alongside the CPU structs. The
   shader now expects precomputed seeds for hue/saturation/brightness override
   noises in addition to the octave seed offsets.
=======
1. **Uniform layout integration** – the WebGPU pipeline now writes the expanded
   `StageUniforms` via `prepareMultiresUniformParams`, but we still need to
   document the packing strategy alongside the CPU structs and audit the
   remaining scalar aliases (`color_params1`, seed offsets, etc.).
>>>>>>> theirs
2. **Octave fidelity** – finish mirroring CPU semantics for alpha-preserving
   layers and additional value distributions. We currently approximate the seed
   sequence (`seed + octave_index - 1`) and still rebuild permutation tables
   inside each invocation; future work should adopt the exact preset-provided
   seeds and reuse cached permutations.
3. **Color workflow polish** – verify the hue/saturation/brightness override
   paths against the CPU implementation (including brightness frequency scaling
   per octave), hook `color_params1` up to any remaining colour controls, and
   support non-simplex distributions for the auxiliary noises.
=======
   expanded `StageUniforms` fields (`color_params0/color_params1`,
   `options0/options1/options2/options3`, `sin_amount`, channel count, etc.)
   and document the packing strategy alongside the CPU structs. The shader now
   expects brightness frequency data in `color_params1.xy` and per-override seed
   offsets in `options3`.
2. **Octave fidelity** – finish mirroring CPU semantics for alpha-preserving
   layers and the broader `ValueDistribution` set. We currently approximate the
   seed sequence (`seed + octave_index - 1`) and still rebuild permutation
   tables inside each invocation; future work should adopt the exact
   preset-provided seeds (one per override noise) and reuse cached permutations.
3. **Color workflow** – verify the new hue/saturation/brightness override paths
   against the Python implementation once uniforms are wired up. Follow-up work
   should add support for the remaining distributions used by overrides
   (center-distance families, row/column indices, etc.) and confirm that the
   pipeline supplies deterministic seeds matching `value.values`.
>>>>>>> theirs
4. **Masking, lattice drift, and normalization** – implement mask sampling,
=======
1. **Octave fidelity** – finish mirroring CPU semantics for alpha-preserving
   layers and additional value distributions. We currently approximate the seed
   sequence (`seed + octave_index - 1`) and still rebuild permutation tables
   inside each invocation; future work should adopt the exact preset-provided
   seeds and reuse cached permutations. The GPU path also clamps the octave
   channel count to four, so revisit temporary alpha channels when parity
   testing.
2. **Color workflow** – hook up hue/saturation/brightness override noises and
   the alternate brightness frequency so HSV modulation matches
   `generators.basic`. `colorParams1` remains unused and should eventually
   carry the extra modulation scalars.
3. **Masking, lattice drift, and normalization** – implement mask sampling,
>>>>>>> theirs
   lattice refract support, and parity for the per-layer/global normalization
   passes (especially the HSV `sin` path, which currently uses a simple
   `map_to_unit`). Preserve staged alpha channels when masks are present.
<<<<<<< ours
4. **Performance tuning** – avoid rebuilding permutation tables per invocation,
   and consider workgroup/shared caching once correctness is locked in.
=======
5. **Pipeline hook-up** – `resolveShaderId` now routes `multires` generators to
   `MULTIRES_WGSL`. Keep this gated behind the remaining TODO items (masks,
   normalization, alternate distributions) before enabling by default in the
   renderer.
6. **Performance tuning** – avoid rebuilding permutation tables per invocation,
   and consider workgroup/shared caching once correctness is locked in.

Leave this file in place until the shader reaches feature parity and is enabled
in the renderer.

Recent progress:

* Stage uniforms are populated on the JS side (`prepareMultiresUniformParams`)
  and the shader is compiled for presets without masks/supersample/AI options.
  Follow up by wiring the remaining colour overrides and validating the
  freq/channel packing against the CPU reference for edge cases (non-square
  shapes, grayscale+alpha, octave alpha combine).
>>>>>>> theirs
