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
1. **Uniform layout integration** – update the WebGPU pipeline to populate the
   expanded `StageUniforms` fields (`brightness_freq`, `options2`, `options3`,
   etc.) and document the packing strategy alongside the CPU structs. The
   shader now expects precomputed seeds for hue/saturation/brightness override
   noises in addition to the octave seed offsets.
2. **Octave fidelity** – finish mirroring CPU semantics for alpha-preserving
   layers and additional value distributions. We currently approximate the seed
   sequence (`seed + octave_index - 1`) and still rebuild permutation tables
   inside each invocation; future work should adopt the exact preset-provided
   seeds and reuse cached permutations.
3. **Color workflow polish** – verify the hue/saturation/brightness override
   paths against the CPU implementation (including brightness frequency scaling
   per octave), hook `color_params1` up to any remaining colour controls, and
   support non-simplex distributions for the auxiliary noises.
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
4. **Performance tuning** – avoid rebuilding permutation tables per invocation,
   and consider workgroup/shared caching once correctness is locked in.
