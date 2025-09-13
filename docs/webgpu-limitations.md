# WebGPU Limitations

Several helper functions in the JavaScript port still rely on synchronous tensor reads. When run under a WebGPU context these helpers will throw an error via `tensor.readSync()`.

The following areas remain CPU/WebGL only:

<<<<<<< ours
- `src/generators.js`: generator array exports
- `src/value.js`: many utilities such as `convolution` and other helpers that still rely on synchronous reads
- `src/effects.js`: most effects that sample tensor data

Utility helpers in `src/util.js`, the colour space conversions in `src/oklab.js`, and the colour helpers in `src/value.js` now use `tensor.read()` and can operate under a WebGPU context.
=======
- `src/value.js`: many utilities such as `normalize` and colour helpers (though some helpers like `clamp01` now use `tensor.read()`)
- `src/effects.js`: most effects that sample tensor data

Utility helpers in `src/util.js`, the generator helpers in `src/generators.js`, and the colour space conversions in `src/oklab.js` now use `tensor.read()` and can operate under a WebGPU context.
>>>>>>> theirs

The remaining functions must be rewritten to use `await tensor.read()` before full WebGPU support is possible.
