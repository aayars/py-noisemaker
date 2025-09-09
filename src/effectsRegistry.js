export const EFFECTS = {};

function getParamNames(fn) {
  const src = fn
    .toString()
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '');
  const match = src.match(/^[^(]*\(([^)]*)\)/);
  if (!match) return [];
  return match[1]
    .split(',')
    .map((p) => p.trim())
    .filter((p) => p.length)
    .map((p) => p.replace(/=.*/, '').trim());
}

export function register(name, fn, defaults = {}) {
  const params = getParamNames(fn);
  const required = ['tensor', 'shape', 'time', 'speed'];
  if (params.length < required.length) {
    throw new Error('Effect functions must accept (tensor, shape, time, speed, ...params)');
  }
  for (let i = 0; i < required.length; i++) {
    if (params[i] !== required[i]) {
      throw new Error(
        `Effect "${name}" must have parameter "${required[i]}" at position ${i + 1}`
      );
    }
  }
  const extraParams = params.slice(required.length);
  const defaultKeys = Object.keys(defaults);
  if (extraParams.length !== defaultKeys.length) {
    throw new Error(
      `Expected ${extraParams.length} default params to "${name}", but got ${defaultKeys.length}`
    );
  }
  for (let i = 0; i < extraParams.length; i++) {
    if (extraParams[i] !== defaultKeys[i]) {
      throw new Error(
        `Parameter "${extraParams[i]}" does not match default key "${defaultKeys[i]}"`
      );
    }
  }
  EFFECTS[name] = { ...defaults, func: fn };
}

export default EFFECTS;
