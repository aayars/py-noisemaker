# WGSL Multires Shader TODO

The multires generator shader now executes per-channel simplex sampling with
octave falloff, reduce-max, and alpha blending paths. It handles grayscale and
RGB/HSV/OKLab workflows, including ridge and per-pixel `sin` modulation, and it
consumes the current pipeline uniforms (`freq`, `speed`, `sin`,
`colorParams0`, `options0`, `options1`). The implementation mirrors the CPU
seed progression (`seed + seed_offset + octave_index - 1`) and converts HSV back
to RGB before storing the results so ping-pong stages see the expected colour
space. Alpha values are preserved for two- and four-channel configurations, and
final writes clamp to the 0–1 range just like the CPU path’s normalization
passes.

Follow-up work before enabling this stage in production:

1. **Masking support** – presets that rely on masks, supersample masks, or
   lattice refract are still routed to the CPU. The shader now mirrors lattice
   drift so only the mask sampling and related auxiliary bindings remain before
   these presets can execute on the GPU path.
2. **Performance** – permutation tables are rebuilt for every pixel and channel,
   matching the CPU algorithm but wasting work on the GPU. Introduce shared
   caching (per workgroup or via uniforms) once correctness is locked in.
3. **Extended distributions** – when pipeline support arrives, add the remaining
   `ValueDistribution` families used by overrides (center-distance, row/column
   indices, etc.) so presets do not fall back to the CPU unexpectedly.

Leave this file in place until the shader reaches feature parity and is enabled
in the WebGPU renderer.
