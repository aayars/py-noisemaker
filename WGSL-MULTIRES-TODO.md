# WGSL Multires Shader TODO

The WGSL module now exposes a richer uniform struct (`StageUniforms`) and
implements multi-octave accumulation for simplex noise, including falloff and
reduce-max blending paths. Alpha blending currently uses a provisional
pre-multiplied mix while we bootstrap mask aware layers. The shader is still not
wired into the runtime. Follow-up work should address the remaining gaps before
enabling the stage:

The shader now lives at `src/webgpu/shaders/multires.wgsl` and is loaded on demand via `src/webgpu/shaders.js` so we can keep runtime bundles lean. Future updates should modify that WGSL module directly.

1. **Uniform layout integration** – update the WebGPU pipeline to populate the
   new `StageUniforms` fields (`options0/options1`, `sin_amount`, channel count,
   etc.) and document the packing strategy alongside the CPU structs.
2. **Octave fidelity** – tighten parity with `generators.multires` by wiring the
   real alpha semantics (mask-preserving layers), honouring non-simplex
   distributions, and mirroring the per-channel/per-octave seed offsets without
   recomputing permutations in every invocation.
3. **Color workflow** – replicate the HSV/OKLab conversion path along with hue,
   saturation, and brightness modulation noise sources (including the optional
   alternate frequency for brightness).
4. **Masking, lattice drift, and sin modulation** – implement mask sampling,
   lattice refract support, and the `sin` parameter. The current shader lacks a
   global normalization pass so we need a strategy that matches the CPU's
   `value.normalize` behaviour.
5. **Pipeline hook-up** – update `resolveShaderId` (and related plumbing) so the
   generator stage resolves to `MULTIRES_WGSL` once feature parity is in place.
6. **Performance tuning** – avoid rebuilding permutation tables per invocation,
   and consider workgroup/shared caching once correctness is locked in.

Leave this file in place until the shader reaches feature parity and is enabled
in the renderer.
