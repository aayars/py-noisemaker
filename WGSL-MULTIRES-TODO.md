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

1. **`sin` parity** – the CPU normalises HSV brightness after the `sin`
   transform using the image-wide min/max range. The shader currently maps the
   per-pixel result back to 0–1 (`map_to_unit`). Revisit this so the GPU path
   mirrors the global remap (probably a reduction pass or a follow-up
   normalisation stage).
2. **Masking and lattice drift** – presets that rely on masks, supersample
   masks, or lattice refract are still routed to the CPU. Once the uniform and
   auxiliary bindings land, add the mask sampling and lattice drift flows so the
   GPU path can participate.
3. **Performance** – permutation tables are rebuilt for every pixel and channel,
   matching the CPU algorithm but wasting work on the GPU. Introduce shared
   caching (per workgroup or via uniforms) once correctness is locked in.
4. **Extended distributions** – when pipeline support arrives, add the remaining
   `ValueDistribution` families used by overrides (center-distance, row/column
   indices, etc.) so presets do not fall back to the CPU unexpectedly.

Leave this file in place until the shader reaches feature parity and is enabled
in the WebGPU renderer.
