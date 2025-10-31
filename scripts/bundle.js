#!/usr/bin/env node
/**
 * Build single-file Noisemaker bundles with esbuild.
 * Produces IIFE, minified, and ESM outputs with presets inlined.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { build } from 'esbuild';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const entryPoint = path.join(repoRoot, 'js', 'noisemaker', 'index.js');
const distDir = path.join(repoRoot, 'dist');
const presetsDslPath = path.join(repoRoot, 'dsl', 'presets.dsl');

if (!fs.existsSync(entryPoint)) {
  console.error(`Bundle entry point not found: ${entryPoint}`);
  process.exit(1);
}

const presetsSource = fs.readFileSync(presetsDslPath, 'utf8');
fs.mkdirSync(distDir, { recursive: true });

const banner = `/**\n * Noisemaker.js - Procedural Noise Generation\n * Bundled on ${new Date().toISOString()}\n */`;

const sharedOptions = {
  entryPoints: [entryPoint],
  bundle: true,
  platform: 'browser',
  target: ['es2020'],
  define: {
    NOISEMAKER_PRESETS_DSL: JSON.stringify(presetsSource),
    __NOISEMAKER_DISABLE_EFFECT_VALIDATION__: 'true',
  },
  legalComments: 'none',
  logLevel: 'warning',
};

async function buildBundle() {
  console.log('Bundling Noisemaker with esbuild...');

  await build({
    ...sharedOptions,
    format: 'iife',
    globalName: 'Noisemaker',
    outfile: path.join(distDir, 'noisemaker.bundle.js'),
    minify: false,
    banner: { js: banner },
  });

  await build({
    ...sharedOptions,
    format: 'iife',
    globalName: 'Noisemaker',
    outfile: path.join(distDir, 'noisemaker.min.js'),
    minify: true,
    banner: { js: banner },
  });

  await build({
    ...sharedOptions,
    format: 'esm',
    outfile: path.join(distDir, 'noisemaker.esm.js'),
    minify: false,
    banner: { js: banner },
  });

  await build({
    ...sharedOptions,
    format: 'cjs',
    outfile: path.join(distDir, 'noisemaker.cjs'),
    minify: false,
    banner: { js: banner },
  });

  console.log('✓ Bundles written to dist/');
  console.log('  - noisemaker.bundle.js');
  console.log('  - noisemaker.min.js');
  console.log('  - noisemaker.esm.js');
  console.log('  - noisemaker.cjs');
}

buildBundle().catch((err) => {
  console.error(err);
  process.exit(1);
});
