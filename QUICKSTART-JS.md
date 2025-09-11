# JavaScript Quickstart

This guide walks through rendering your first image with the experimental Noisemaker JavaScript API.

## Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/aayars/py-noisemaker
cd py-noisemaker
npm install
```

## Render your first preset

Create an HTML file that imports the source modules directly:

```html
<!doctype html>
<canvas id="noise"></canvas>
<script type="module">
  import { Context } from './src/context.js';
  import { render } from './src/composer.js';
  import { savePNG } from './src/util.js';

  const canvas = document.getElementById('noise');
  const ctx = new Context(canvas);

  // Render the built-in "basic" preset with a fixed seed
  const tensor = render('basic', 42, { width: 256, height: 256, ctx });

  // Optionally download the result
  // savePNG(tensor, 'noise.png');
</script>
```

Serve this file via a local web server (for example, `python -m http.server`) before opening it in your browser.

The `render` function writes directly to the supplied canvas and returns a `Tensor` containing the raw pixel data. When running under Node or without a canvas, use `ctx: new Context(null)` and inspect the tensor via `tensor.read()`.

## Run the test suite

```bash
npm test
```

The test suite invokes the Python reference implementation at runtime and
compares outputs directly—no fixtures are involved. Do not modify the Python
reference or disable tests; discrepancies should surface as test failures.
