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

