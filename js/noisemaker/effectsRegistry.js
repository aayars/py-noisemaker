/**
 * Mapping of effect names to their implementation and default parameters.
 * The registry is populated via {@link register} and is primarily consumed by
 * the composer and test suites.
 *
 * Each entry has the shape `{ ...defaults, func }` where `defaults` contains
 * the optional parameters for the effect and `func` is the callback that will
 * perform the mutation on the tensor.
 *
 * @type {Record<string, object>}
 */
export const EFFECTS = {};

/**
 * Metadata describing the default parameters for each registered effect. This
 * mirrors {@link EFFECTS} without the execution callback and is intended for
 * introspection or documentation generation.
 *
 * @type {Record<string, object>}
 */
export const EFFECT_METADATA = {};

/**
 * List all registered effect names.
 *
 * @returns {string[]} Array of effect identifiers.
 */
export function list() {
  return Object.keys(EFFECTS);
}

function getParamNames(fn) {
  const src = fn
    .toString()
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/.*$/gm, "");
  const match = src.match(/^[^(]*\(([^)]*)\)/);
  if (!match) return [];
  return match[1]
    .split(",")
    .map((p) => p.trim())
    .filter((p) => p.length)
    .map((p) => p.replace(/=.*/, "").trim());
}

/**
 * Register a new effect.
 *
 * ```javascript
 * import { register } from "./effectsRegistry.js";
 *
 * function warp(tensor, shape, time, speed, freq = 2) {
 *   // ...effect implementation...
 * }
 *
 * register("warp", warp, { freq: 2 });
 * ```
 *
 * Effect functions must accept `(tensor, shape, time, speed, ...params)` in
 * that exact order. Any additional parameters must provide a corresponding
 * default value in the `defaults` object.
 *
 * @param {string} name Unique effect identifier.
 * @param {Function} fn Effect callback invoked during rendering.
 * @param {object} [defaults={}] Map of optional parameter names to default values.
 */
const VALIDATE_EFFECT_SIGNATURES =
  typeof __NOISEMAKER_DISABLE_EFFECT_VALIDATION__ === 'undefined';

export function register(name, fn, defaults = {}) {
  const params = getParamNames(fn);
  const required = ["tensor", "shape", "time", "speed"];
  if (VALIDATE_EFFECT_SIGNATURES) {
    if (params.length < required.length) {
      throw new Error(
        "Effect functions must accept (tensor, shape, time, speed, ...params)",
      );
    }
    for (let i = 0; i < required.length; i++) {
      if (params[i] !== required[i]) {
        throw new Error(
          `Effect "${name}" must have parameter "${required[i]}" at position ${i + 1}`,
        );
      }
    }
    const extraParams = params.slice(required.length);
    const defaultKeys = Object.keys(defaults);
    if (extraParams.length !== defaultKeys.length) {
      throw new Error(
        `Expected ${extraParams.length} default params to "${name}", but got ${defaultKeys.length}`,
      );
    }
    for (let i = 0; i < extraParams.length; i++) {
      if (extraParams[i] !== defaultKeys[i]) {
        throw new Error(
          `Parameter "${extraParams[i]}" does not match default key "${defaultKeys[i]}"`,
        );
      }
    }
  }
  EFFECT_METADATA[name] = { ...defaults };
  EFFECTS[name] = { ...defaults, func: fn };
}
EFFECTS.list = list;

export default EFFECTS;
