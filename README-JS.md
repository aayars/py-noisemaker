# Noisemaker JavaScript Port

This document is for the experimental JS port of Noisemaker. See additional [porter's notes](VANILLA_JS_PORT_SPEC.md). For a step-by-step introduction, read [QUICKSTART-JS.md](QUICKSTART-JS.md).

The project now relies on a common **3‑D OpenSimplex** implementation across Python and JavaScript.

## Cross-language parity tests

`npm test` runs the JavaScript suite, which invokes the Python reference
implementation in a subprocess and compares outputs directly.  No fixture files
or canned images are used.  Any difference between languages is treated as a
test failure—do not modify the Python reference implementation, and do not skip
or weaken tests to hide problems.

## Vanilla JS effects registry

The Vanilla JavaScript port includes an `effectsRegistry` helper that tracks all
available post-processing effects along with their default parameters. After an
effect is registered the defaults can be inspected via `EFFECT_METADATA`.

```javascript
import { register, EFFECT_METADATA } from "./src/effectsRegistry.js";

function ripple(tensor, shape, time, speed, amount = 1.0) {
  // ...effect implementation...
}

register("ripple", ripple, { amount: 1.0 });

console.log(EFFECT_METADATA.ripple); // => { amount: 1.0 }
```

Effect callbacks must accept `(tensor, shape, time, speed, ...params)` in that
order. Any additional parameters require matching default values supplied in the
`defaults` object passed to `register`.

## WebGPU acceleration

Noisemaker can offload some Voronoi calculations to WebGPU when a compatible
GPU is available. The following diagram types are accelerated:

- `range`
- `color_range`
- `regions`
- `color_regions`
- `range_regions`
- `flow`
- `color_flow`

Voronoi shaders also support a subset of distance metrics on the GPU:

- `euclidean`
- `manhattan`
- `chebyshev`
- `octagram`
- `triangular`
- `hexagram`
- `sdf` (with `sdfSides` ≥ 3)

Other diagram types, distance metrics, or values of `nth` ≥ 64 automatically
fall back to the CPU implementation to maintain feature parity.

### Enabling WebGPU

WebGPU is still an experimental API. To try GPU acceleration:

- **Chrome/Edge** – enable the *WebGPU* flag at `chrome://flags` or start the
  browser with `--enable-unsafe-webgpu`.
- **Safari** – enable *WebGPU* under *Develop → Experimental Features*.
- **Firefox Nightly** – set `dom.webgpu.enabled` to `true` in `about:config` and
  launch with `--enable-features=webgpu`.

WebGPU only works in secure contexts (HTTPS or `localhost`). If
`navigator.gpu` is undefined, the API is unavailable or disabled.

### Troubleshooting initialization

- **`navigator.gpu` is undefined** – verify WebGPU is enabled and you're using a
  recent browser version in a secure context.
- **`requestAdapter` or `requestDevice` fails** – update graphics drivers or try
  a different browser.
- **Validation errors** – check the browser console for details; the demo will
  automatically fall back to WebGL when initialization fails.

