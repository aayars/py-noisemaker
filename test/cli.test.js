import assert from 'assert';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const cliPath = path.resolve(__dirname, '../js/bin/noisemaker-js');
const outputPath = path.resolve(__dirname, 'tmp-cli-output.png');
const applyInputPath = path.resolve(__dirname, 'tmp-cli-input.png');
const applyOutputPath = path.resolve(__dirname, 'tmp-cli-apply-output.png');

for (const file of [outputPath, applyInputPath, applyOutputPath]) {
  try {
    fs.unlinkSync(file);
  } catch (_) {
    /* ignore */
  }
}

const listGenerate = spawnSync('node', [cliPath, 'generate', '--help-presets'], {
  encoding: 'utf8',
});

if (listGenerate.status !== 0) {
  const stderr = listGenerate.stderr ? `\n${listGenerate.stderr}` : '';
  const stdout = listGenerate.stdout ? `\n${listGenerate.stdout}` : '';
  throw new Error(`CLI help-presets generate exited with status ${listGenerate.status}${stderr}${stdout}`);
}

assert.ok(listGenerate.stdout.includes('Available generator presets:'), 'Missing generator preset list header');

const listApply = spawnSync('node', [cliPath, 'apply', '--help-presets'], {
  encoding: 'utf8',
});

if (listApply.status !== 0) {
  const stderr = listApply.stderr ? `\n${listApply.stderr}` : '';
  const stdout = listApply.stdout ? `\n${listApply.stdout}` : '';
  throw new Error(`CLI help-presets apply exited with status ${listApply.status}${stderr}${stdout}`);
}

assert.ok(listApply.stdout.includes('Available effect presets:'), 'Missing effect preset list header');

const result = spawnSync('node', [cliPath, 'generate', 'basic', '--filename', outputPath, '--width', '32', '--height', '32', '--seed', '1'], {
  encoding: 'utf8',
});

if (result.status !== 0) {
  const stderr = result.stderr ? `\n${result.stderr}` : '';
  const stdout = result.stdout ? `\n${result.stdout}` : '';
  throw new Error(`CLI exited with status ${result.status}${stderr}${stdout}`);
}

assert.ok(fs.existsSync(outputPath), 'Output file was not created');
const stats = fs.statSync(outputPath);
assert.ok(stats.size > 0, 'Output file is empty');

fs.unlinkSync(outputPath);

const png = new PNG({ width: 8, height: 8 });
for (let y = 0; y < png.height; y += 1) {
  for (let x = 0; x < png.width; x += 1) {
    const idx = (y * png.width + x) * 4;
    png.data[idx] = x * 32;
    png.data[idx + 1] = y * 32;
    png.data[idx + 2] = 128;
    png.data[idx + 3] = 255;
  }
}
const inputBuffer = PNG.sync.write(png);
fs.writeFileSync(applyInputPath, inputBuffer);

const applyResult = spawnSync('node', [cliPath, 'apply', 'aberration', applyInputPath, '--filename', applyOutputPath, '--seed', '1', '--no-resize'], {
  encoding: 'utf8',
});

if (applyResult.status !== 0) {
  const stderr = applyResult.stderr ? `\n${applyResult.stderr}` : '';
  const stdout = applyResult.stdout ? `\n${applyResult.stdout}` : '';
  throw new Error(`CLI apply exited with status ${applyResult.status}${stderr}${stdout}`);
}

assert.ok(fs.existsSync(applyOutputPath), 'Apply output file was not created');
const applyStats = fs.statSync(applyOutputPath);
assert.ok(applyStats.size > 0, 'Apply output file is empty');

fs.unlinkSync(applyInputPath);
fs.unlinkSync(applyOutputPath);

console.log('cli tests passed');
