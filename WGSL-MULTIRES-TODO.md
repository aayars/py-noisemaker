# WGSL Multires Shader TODO

The WGSL module now exposes a richer uniform struct (`StageUniforms`) and
implements multi-octave accumulation for simplex noise, including falloff and
reduce-max blending paths. Alpha blending currently uses a provisional
pre-multiplied mix while we bootstrap mask aware layers. The shader now
evaluates per-channel simplex seeds, performs HSV/OKLab conversion, applies the
HSV hue/saturation/brightness overrides (including brightness frequency
overrides), and honours reduce-max/alpha octave blending, but it is still not
wired into the runtime. Follow-up work should address the remaining gaps before
enabling the stage:

The shader now lives at `src/webgpu/shaders/multires.wgsl` and is loaded on demand via `src/webgpu/shaders.js` so we can keep runtime bundles lean. Future updates should modify that WGSL module directly.

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
   lattice refract support, and parity for the per-layer/global normalization
   passes (especially the HSV `sin` path, which currently uses a simple
   `map_to_unit`). Preserve staged alpha channels when masks are present.
5. **Pipeline hook-up** – update `resolveShaderId` (and related plumbing) so the
   generator stage resolves to `MULTIRES_WGSL` once feature parity is in place.
6. **Performance tuning** – avoid rebuilding permutation tables per invocation,
   and consider workgroup/shared caching once correctness is locked in.

Leave this file in place until the shader reaches feature parity and is enabled
in the renderer.
