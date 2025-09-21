# WGSL Multires Shader TODO

This commit introduces the foundational WGSL module for the `multires` generator,
including the seeded permutation builder and OpenSimplex 2D/3D routines. The
compute entry point currently renders a single-octave grayscale output and is
not yet wired into the WebGPU pipeline. Follow-up work should address the
remaining pieces before enabling the stage:

1. **Uniform layout integration** – teach the pipeline to populate the stage
   uniforms expected by the shader (frequency, octave parameters, distribution
   flags, etc.) or specialize the WGSL struct per preset.
2. **Octave accumulation** – port the full multi-octave blending logic (including
   the `OctaveBlending` modes and alpha handling) and ensure the shader mirrors
   `generators.multires`.
3. **Color workflow** – replicate the HSV/OKLab conversion path along with hue,
   saturation, and brightness modulation noise sources.
4. **Masking and lattice drift** – implement the optional mask resources and the
   lattice refract step used by the CPU version.
5. **Pipeline hook-up** – update `resolveShaderId` (and related plumbing) so the
   generator stage resolves to `MULTIRES_WGSL` once the shader reaches parity.
6. **Performance tuning** – consider caching the permutation tables per dispatch
   (e.g., workgroup-shared state) to avoid recomputing them per invocation once
   correctness is locked in.

Leave this file in place until the shader reaches feature parity and is enabled
in the renderer.
