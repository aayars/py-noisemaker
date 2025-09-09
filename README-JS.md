# Noisemaker JavaScript Port

This document is for the experimental JS port of Noisemaker. See additional [porter's notes](VANILLA_JS_PORT_SPEC.md).

## Updating JavaScript test fixtures
 
The JavaScript tests use precomputed outputs from the Python reference implementation. If you change any algorithms that affect these expectations, regenerate the fixture data:

```bash
python scripts/generate_fixtures.py
```

This script rewrites the JSON files in `test/fixtures/`. Commit the updated fixtures along with your code changes.

## Cross-language image regression

To verify that the Python and JavaScript implementations stay in sync, generate
small simplex-noise images for a set of seeds and compare the pixel values:

```bash
npm run image-regress
```

Baseline PNGs are written to `test/image-fixtures/`. Divergences are reported
with the offending seed to help catch regressions.

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

