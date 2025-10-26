# Python ↔ JavaScript Parity Specification

This document defines requirements for bringing the Python and JavaScript implementations of Noisemaker into deterministic lockstep.  The goal is that a seed, preset and parameter set produce the same intermediate values and the same final pixels in both languages.

Python remains the authoritative reference; the JavaScript port must match its behaviour.

All parity tests under `test/parity/` **must** execute the Python and
JavaScript implementations side by side and compare their results directly.
No precomputed or hand-edited fixtures are permitted.  The tests invoke the
JavaScript reference at runtime and assert a literal 1:1 match against the
Python output (see `test/parity/test_generators.py` and
`test/parity/test_effects.py` for examples).  The Python reference
implementation may not be altered to make tests pass, and tests must not be
removed, skipped, or otherwise weakened. Any divergence between languages
must surface as a test failure rather than being masked. You may not use
canned return values or synthetic results. All results need to be based on
actual returned values from the functions being tested, and if the test
is failing, you need to fix the root cause.

All 2D tests are to be based on an image shape of 128x128x3.

---

## 1. Scope and Goals

* **Determinism** – every source of randomness must be reproducible and yield identical sequences across languages.
* **Unit‑level cohesion** – low level building blocks (RNG, simplex gradients, value noise, etc.) must return the same values.
* **Pixel parity** – canonical presets rendered in both languages should compare bit‑for‑bit (or within a single‑byte tolerance when unavoidable).
* **Automation** – parity tests spawn the JavaScript implementation at runtime
  and compare against live output; no stored fixtures or manual editing.

---

## 2. Random Number Generation

1. **Single PRNG** – adopt one 32‑bit PRNG (Mulberry32 or equivalent) implemented in both languages.
2. **Seed handling**
   * Accept an unsigned 32‑bit seed.
   * Normalise the seed in the same way on both sides (e.g. mask with `0xffffffff`).
   * Document endianness and integer overflow behaviour.
3. **API**
   * Expose `setSeed`, `getSeed`, `random() → [0,1)`, `randomInt(min,max)`, `choice(list)`.
   * All higher‑level modules must use this API; direct calls to language standard RNGs are forbidden.
4. **Parity tests**
   * Python tests spawn the JavaScript RNG and stream its output for shared
     seeds, asserting equality to within `1e‑9`.
   * No fixture files are written or consumed; comparisons are performed in
     real time to avoid drift.

---

## 3. Simplex Noise

1. **Algorithm**
   * Implement the same 3‑D OpenSimplex variant. 4‑D noise is intentionally unsupported.
   * Gradient table and permutation arrays are generated at runtime using the shared RNG; no fixtures are stored.
2. **Seeding**
   * Seeding must only call the RNG in a documented order; consuming more or fewer values is a breaking change.
3. **Unit tests**
   * Deterministic tests sample a grid `[(0…3)/7]³` for several seeds and assert value equality within `1e‑6`.
   * Tests include checks for gradients and derivatives to ensure orientation parity and loopability.
   * A unified integration test invokes the JS implementation from Python and asserts matching grids, derivatives, and tiles for shared seeds within `1e-6`.
4. **Coverage**
   * Tests must include normal, loopable, and tiled noise helpers.

---

## 4. Higher‑level Modules

Modules that consume randomness (value noise, points, masks, effects, composer, presets) must:

* Call RNG through the shared API only.
* Document the number and order of RNG calls so the execution graphs remain aligned.
* Provide parity tests that invoke the JavaScript implementation and compare
  outputs directly with the Python results.  See `test/parity/test_generators.py`
  and `test/parity/test_effects.py` for the required structure.

---

## 5. Execution Graph Consistency

1. **Preset graph description** – document, in `docs/exec-graph.md`, the canonical order of effect application and RNG consumption for each preset.
2. **Instrumentation**
   * Provide a debug mode that logs RNG calls and effect execution order.
   * Parity tests compare these logs for Python vs. JS.
3. **Change detection**
   * Any change in call order or RNG usage must fail tests.  Do not hide
     failures by altering or skipping tests.

---

## 6. Pixel‑level Verification

1. **Runtime renders**
   * Parity tests render canonical presets in both languages at runtime and compare the resulting pixel tensors directly.
2. **Comparison**
   * Comparison is byte‑exact; if floating‑point differences appear, allow tolerance ≤1/255 per channel and report mean/maximum error.
3. **Continuous integration**
   * CI pipelines run these comparisons on both sides to guarantee no regressions.

---

## 7. Runtime Comparison Workflow

* No fixture files are versioned in the repository.
* Python tests invoke the JavaScript reference implementation during each run.
* `npm test` and `pytest` must fail if the outputs diverge.
* When behaviour intentionally changes, update both implementations so that parity tests pass without disabling or weakening them.

---

## 8. Test Matrix

| Area      | Python test                       | JS test                    |
|-----------|-----------------------------------|----------------------------|
| RNG       | `test/parity/test_rng.py`         | `test/rng.test.js`         |
| Simplex   | `test/parity/test_simplex.py`     | `test/simplex.test.js`     |
| Value     | `test/parity/test_value.py`       | `test/value.test.js`       |
| Points    | `test/parity/test_points.py`      | `test/points.test.js`      |
| Effects   | `test/parity/test_effects.py`     | `test/effects.test.js`     |
| Composer  | `test/parity/test_composer.py`    | `test/composer.test.js`    |
| Renders   | `test/parity/test_renders.py`     | `test/renders.test.js`     |

All parity tests must run in CI for both languages.

---

## 9. Development Workflow

1. Implement feature or bug fix in Python.
2. Ensure all Python tests, including parity tests, pass.
3. Port identical logic to JS.
4. Run JS tests which invoke the Python reference implementation; resolve any divergences.
5. Commit code and spec updates in the same pull request.

---

## 10. Open Issues / Questions

* Floating‑point differences may still occur on different hardware. Investigate fixed‑point alternatives if required.
* Document any intentional deviations (e.g., performance shortcuts) in the spec and ensure tests reflect the intended behaviour.
