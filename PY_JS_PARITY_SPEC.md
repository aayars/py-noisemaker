# Python ↔ JavaScript Parity Specification

This document defines requirements for bringing the Python and JavaScript implementations of Noisemaker into deterministic lockstep.  The goal is that a seed, preset and parameter set produce the same intermediate values and the same final pixels in both languages.

Python remains the authoritative reference; the JavaScript port must match its behaviour.

---

## 1. Scope and Goals

* **Determinism** – every source of randomness must be reproducible and yield identical sequences across languages.
* **Unit‑level cohesion** – low level building blocks (RNG, simplex gradients, value noise, etc.) must return the same values.
* **Pixel parity** – canonical presets rendered in both languages should compare bit‑for‑bit (or within a single‑byte tolerance when unavoidable).
* **Automation** – fixtures and tests must be generated and consumed automatically; no manual editing.

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
   * Generate fixture files: `fixtures/rng/seed_<n>.json` containing the first 1 000 values from Python.
   * JS unit test consumes these fixtures and asserts equality to within `1e‑9`.
   * Add a mirror test in Python that loads the same fixture and self‑verifies, preventing drift.

---

## 3. Simplex Noise

1. **Algorithm**
   * Implement the same 3‑D OpenSimplex variant.
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
* Provide unit fixtures verifying critical outputs. Examples:
  * `value` – first 64 samples of `value_noise()` at seed 1.
  * `points` – coordinates produced by `cloud_points()` for fixed count and seed.
  * `effects` – parameter tensors for `worms`, `octave_blur`, etc.
* JS tests mirror the Python fixtures; Python has self‑tests to detect fixture drift.

---

## 5. Execution Graph Consistency

1. **Preset graph description** – document, in `docs/exec-graph.md`, the canonical order of effect application and RNG consumption for each preset.
2. **Instrumentation**
   * Provide a debug mode that logs RNG calls and effect execution order.
   * Parity tests compare these logs for Python vs. JS.
3. **Change detection**
   * A change in call order or RNG usage must fail tests unless the fixtures are intentionally regenerated.

---

## 6. Pixel‑level Verification

1. **Reference renders**
   * Python script `scripts/generate_fixtures.py` renders a set of canonical presets (e.g., `basic`, `worms`, `voronoi`) at 128×128 with known seeds and writes PNGs under `fixtures/renders/`.
2. **Comparison**
   * JS tests render the same presets using Canvas/WebGL and compare against reference images.
   * Comparison is byte‑exact; if floating‑point differences appear, allow tolerance ≤1/255 per channel and report mean/maximum error.
3. **Continuous integration**
   * CI pipelines run these comparisons on both sides to guarantee no regressions.

---

## 7. Fixture Workflow

* Fixtures reside in `fixtures/<category>/` and are versioned with the repo.
* `scripts/generate_fixtures.py` regenerates all fixtures from the Python implementation.
* `npm test` and `pytest` automatically load fixtures; tests fail if fixtures are missing or outdated.
* When behaviour intentionally changes, regenerate fixtures and commit alongside code changes.

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
2. Regenerate fixtures and update Python tests.
3. Port identical logic to JS.
4. Run JS unit tests against fixtures; update fixtures only if both implementations agree.
5. Commit code + fixtures + spec updates in the same pull request.

---

## 10. Open Issues / Questions

* Floating‑point differences may still occur on different hardware. Investigate fixed‑point alternatives if required.
* Document any intentional deviations (e.g., performance shortcuts) in the spec and ensure tests reflect the intended behaviour.