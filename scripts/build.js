import fs from 'fs';
import path from 'path';

const modules = [
  'constants.js',
  'simplex.js',
  'value.js',
  'points.js',
  'masks.js',
  'effectsRegistry.js',
  'effects.js',
  'composer.js',
  'palettes.js',
  'presets.js',
  'util.js',
  'oklab.js',
  'glyphs.js',
  'tensor.js',
  'context.js'
];

const distDir = path.resolve('dist');
fs.mkdirSync(distDir, { recursive: true });

// ES module bundle that re-exports individual modules
const esm = modules.map(m => `export * from '../src/${m}';`).join('\n');
fs.writeFileSync(path.join(distDir, 'noisemaker.mjs'), esm);

// UMD wrapper that dynamically imports the ES module
const umd = `(function(root, factory){\n` +
`  if (typeof module === 'object' && module.exports) {\n` +
`    module.exports = factory();\n` +
`  } else {\n` +
`    root.Noisemaker = factory();\n` +
`  }\n` +
`}(this, function(){\n` +
`  return import('./noisemaker.mjs');\n` +
`}));\n`;
fs.writeFileSync(path.join(distDir, 'noisemaker.umd.js'), umd);

console.log('Built ES module and UMD bundles to dist/');
