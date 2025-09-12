# WebGPU Limitations

Several helper functions in the JavaScript port still rely on synchronous tensor reads. When run under a WebGPU context these helpers will throw an error via `tensor.readSync()`.

The following areas remain CPU/WebGL only:

- `src/util.js`: `savePNG`, `fromSRGB`, `toSRGB`
- `src/oklab.js`: colour space conversions
- `src/composer.js`: array drawing helpers
- `src/generators.js`: generator array exports
- `src/value.js`: many utilities such as `normalize` and colour helpers
- `src/effects.js`: most effects that sample tensor data

These functions must be rewritten to use `await tensor.read()` before full WebGPU support is possible.
