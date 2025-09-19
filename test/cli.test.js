import assert from 'assert';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const cliPath = path.resolve(__dirname, '../bin/noisemaker-js');
const outputPath = path.resolve(__dirname, 'tmp-cli-output.png');

try {
  fs.unlinkSync(outputPath);
} catch (_) {
  /* ignore */
}

const result = spawnSync('node', [cliPath, 'generate', 'basic', '--filename', outputPath, '--width', '32', '--height', '32', '--seed', '0'], {
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

console.log('cli tests passed');
