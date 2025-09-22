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
passes. Value distributions now mirror the CPU implementation for simplex,
exponential, constant, column/row index, and centre-distance families so octave
generation stays on the GPU for those presets.

Follow-up work before enabling this stage in production:

1. **Procedural masks** – the GPU path now samples static masks by uploading a
   per-octave mask atlas, but procedural masks (those backed by functions in
   `masks.js`) still fall back to the CPU. Port the procedural generators so the
   WebGPU path can animate dropout/truchet/etc. without CPU assistance. Static
   mask uploads are now cached per descriptor/resolution so procedural support
   should integrate with the new cache rather than recomputing every frame.

Leave this file in place until the shader reaches feature parity and is enabled
in the WebGPU renderer.
