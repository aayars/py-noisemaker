# Noisemaker JavaScript Port

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

