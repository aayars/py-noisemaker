# JavaScript Quickstart

This guide walks through rendering your first image with the experimental Noisemaker JavaScript API.

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
npm install
```

2. Build the distributable bundles:

```bash
npm run build
```

This writes `dist/noisemaker.mjs` (ES module) and `dist/noisemaker.umd.js` (UMD) for use in browsers or other bundlers.

## Render your first preset

Create an HTML file and import the built library:

```html
<!doctype html>
<canvas id="noise"></canvas>
<script type="module">
  import { Context, render, savePNG } from './dist/noisemaker.mjs';

  const canvas = document.getElementById('noise');
  const ctx = new Context(canvas);

  // Render the built-in "basic" preset with a fixed seed
  const tensor = render('basic', 42, { width: 256, height: 256, ctx });

  // Optionally download the result
  // savePNG(tensor, 'noise.png');
</script>
```

The `render` function writes directly to the supplied canvas and returns a `Tensor` containing the raw pixel data. When running under Node or without a canvas, use `ctx: new Context(null)` and inspect the tensor via `tensor.read()`.

## Run the test suite

```bash
npm test
```

The tests compare the JavaScript port against the Python reference implementation to catch regressions.
