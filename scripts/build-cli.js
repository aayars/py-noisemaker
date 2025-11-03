#!/usr/bin/env node
/**
 * Build a self-contained Noisemaker CLI bundle for Node environments.
 * Produces an executable script and Windows shim under dist/cli/.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const entryPoint = path.join(repoRoot, 'js', 'bin', 'noisemaker-js');
const cliDistDir = path.join(repoRoot, 'dist', 'cli');
const presetsDslPath = path.join(repoRoot, 'dsl', 'presets.dsl');

if (!fs.existsSync(entryPoint)) {
  console.error(`Noisemaker CLI entry not found at ${entryPoint}`);
  process.exit(1);
}

if (!fs.existsSync(presetsDslPath)) {
  console.error(`Preset DSL source missing at ${presetsDslPath}`);
  process.exit(1);
}

const presetsSource = fs.readFileSync(presetsDslPath, 'utf8');
fs.mkdirSync(cliDistDir, { recursive: true });

// Note: No shebang banner for SEA bundles - they don't execute as scripts
const banner = [
  '/*',
  ' * Noisemaker CLI bundle',
  ` * Built on ${new Date().toISOString()}`,
  ' */',
].join('\n');

const bundleOut = path.join(cliDistDir, 'noisemaker-bundle.cjs');

await build({
  entryPoints: [entryPoint],
  bundle: true,
  platform: 'node',
  target: ['node20'],
  format: 'cjs',
  outfile: bundleOut,
  banner: { js: banner },
  define: {
    __NOISEMAKER_DISABLE_EFFECT_VALIDATION__: 'true',
    // Define placeholder for DSL - will be replaced with actual embedded DSL after build
    NOISEMAKER_PRESETS_DSL: '"__PLACEHOLDER__"',
  },
  legalComments: 'none',
  logLevel: 'info',
  logOverride: {
    'empty-import-meta': 'silent',
  },
});

// Embed presets DSL into bundle for SEA
let bundleCode = fs.readFileSync(bundleOut, 'utf-8');

// Strip any shebang that esbuild might have preserved from entry point
bundleCode = bundleCode.replace(/^#!.*\n/, '');

const encoded = Buffer.from(presetsSource).toString('base64');
const wrappedCode = `process.env.NOISEMAKER_EMBEDDED_DSL = '${encoded}';\n${bundleCode}`;
fs.writeFileSync(bundleOut, wrappedCode, 'utf-8');

console.log('✓ Built Noisemaker CLI bundle with embedded DSL in dist/cli/');
