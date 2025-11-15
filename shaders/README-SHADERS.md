# Noisemaker Shaders

WebGPU/WGSL implementations of selected Noisemaker effects. This directory is a distinct, standalone track separate from the Python and JavaScript implementations—changes here do not affect the Python/JS pipelines, and vice‑versa.

## Current status

- Effects should load and run in the viewer without console errors.
- Many effects are still prototypes; quality and parameters may evolve.
- Visual parity with the Python reference is an active work in progress.

## Using the viewer

Open `demo/gpu-effects/index.html` with the project’s development server and select an effect from the menu. Each effect exposes parameters that mirror the Python reference where practical.

## Development notes

- This shader collection is independent from the Python and JS pipelines. Keep implementations and controls consistent, but do not couple code across directories.
- All textures/buffers are treated as 4‑channel RGBA. Do not branch on channel count.
- WGSL struct members end with a trailing comma, not a semicolon.
- Controls in the demo should strive to match Python effect params (except "shape").
- See `shaders/IMPLEMENTATION_GUIDE.md` for architecture, binding layouts, and tutorials.
