# JavaScript Quickstart

This guide walks through rendering your first image with the experimental Noisemaker JavaScript API.

## Setup

<<<<<<< ours
1. Clone the repository and install dependencies:
=======
1. Clone the repository and install dependencies (for tests):
>>>>>>> theirs

```bash
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
npm install
```

<<<<<<< ours
2. Build the distributable bundles:

```bash
npm run build
```

This writes `dist/noisemaker.mjs` (ES module) and `dist/noisemaker.umd.js` (UMD) for use in browsers or other bundlers.

## Render your first preset

Create an HTML file and import the built library:
=======
## Render your first preset

Create an HTML file that imports the source modules directly:
>>>>>>> theirs

```html
<!doctype html>
<canvas id="noise"></canvas>
<script type="module">
<<<<<<< ours
  import { Context, render, savePNG } from './dist/noisemaker.mjs';
=======
  import { Context } from './src/context.js';
  import { render } from './src/composer.js';
  import { savePNG } from './src/util.js';
>>>>>>> theirs

  const canvas = document.getElementById('noise');
  const ctx = new Context(canvas);

  // Render the built-in "basic" preset with a fixed seed
  const tensor = render('basic', 42, { width: 256, height: 256, ctx });

  // Optionally download the result
  // savePNG(tensor, 'noise.png');
</script>
```

<<<<<<< ours
=======
Serve this file via a local web server (for example, `python -m http.server`) before opening it in your browser.

>>>>>>> theirs
The `render` function writes directly to the supplied canvas and returns a `Tensor` containing the raw pixel data. When running under Node or without a canvas, use `ctx: new Context(null)` and inspect the tensor via `tensor.read()`.

## Run the test suite

```bash
npm test
```

The tests compare the JavaScript port against the Python reference implementation to catch regressions.
