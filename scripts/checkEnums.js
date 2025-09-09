#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import fs from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function getPythonEnums() {
  const py = `
import json, enum
import noisemaker.constants as c
result = {}
for name, obj in vars(c).items():
    if name != 'Enum' and isinstance(obj, enum.EnumMeta):
        result[name] = {m.name: m.value for m in obj}
print(json.dumps(result))
`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    console.error(res.stderr);
    process.exit(1);
  }
  return JSON.parse(res.stdout);
}

function getPythonMasks() {
  const py = `
import json
from noisemaker.masks import Masks, mask_shape
result = {}
for mask, value in Masks.items():
    if not callable(value):
        result[mask.name] = mask_shape(mask)
print(json.dumps(result))
`;
  const res = spawnSync('python', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  if (res.status !== 0) {
    console.error(res.stderr);
    process.exit(1);
  }
  return JSON.parse(res.stdout);
}

async function getJsEnums() {
  const mod = await import(resolve(repoRoot, 'src/constants.js'));
  const enums = {};
  for (const [key, value] of Object.entries(mod)) {
    if (
      value &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      Object.values(value).every((v) => typeof v === 'number')
    ) {
      enums[key] = value;
    }
  }
  return enums;
}

async function getJsMasks() {
  const constMod = await import(resolve(repoRoot, 'src/constants.js'));
  const { Masks } = await import(resolve(repoRoot, 'src/masks.js'));
  const masks = {};
  for (const [name, value] of Object.entries(constMod.ValueMask || {})) {
    const entry = Masks[value];
    if (entry && typeof entry !== 'function') {
      const height = entry.length;
      const width = entry[0].length;
      const channels = Array.isArray(entry[0][0]) ? entry[0][0].length : 1;
      masks[name] = [height, width, channels];
    }
  }
  return masks;
}

function compare(pyEnums, jsEnums) {
  const mismatches = [];
  for (const [name, pyEnum] of Object.entries(pyEnums)) {
    const jsEnum = jsEnums[name];
    if (!jsEnum) {
      mismatches.push(`Missing enum ${name} in JS`);
      continue;
    }
    for (const [member, value] of Object.entries(pyEnum)) {
      if (jsEnum[member] !== value) {
        mismatches.push(
          `Enum ${name} differs for ${member}: python=${value} js=${jsEnum[member]}`
        );
      }
    }
    for (const member of Object.keys(jsEnum)) {
      if (!(member in pyEnum)) {
        mismatches.push(`Extra member ${name}.${member} in JS`);
      }
    }
  }
  for (const name of Object.keys(jsEnums)) {
    if (!(name in pyEnums)) {
      mismatches.push(`Extra enum ${name} in JS`);
    }
  }
  return mismatches;
}

function compareMasks(pyMasks, jsMasks) {
  const mismatches = [];
  for (const [name, jsShape] of Object.entries(jsMasks)) {
    const pyShape = pyMasks[name];
    if (!pyShape) {
      mismatches.push(`Missing mask ${name} in Python`);
      continue;
    }
    if (JSON.stringify(pyShape) !== JSON.stringify(jsShape)) {
      mismatches.push(`Mask ${name} shape differs: python=${pyShape} js=${jsShape}`);
    }
  }
  return mismatches;
}

function generateEnums(pyEnums) {
  const lines = ['// Auto-generated enumeration maps'];
  for (const name of Object.keys(pyEnums)) {
    const members = pyEnums[name];
    lines.push(`export const ${name} = Object.freeze({`);
    for (const [k, v] of Object.entries(members)) {
      lines.push(`  ${k}: ${v},`);
    }
    lines.push('});', '');
  }
  return lines.join('\n');
}

async function main() {
  const update = process.argv.includes('--update');
  const pyEnums = getPythonEnums();
  const jsEnums = await getJsEnums();
  const pyMasks = getPythonMasks();
  const jsMasks = await getJsMasks();
  const mismatches = [...compare(pyEnums, jsEnums), ...compareMasks(pyMasks, jsMasks)];

  if (mismatches.length) {
    console.error('Enum/mask mismatches:\n' + mismatches.join('\n'));
    if (!update) {
      process.exit(1);
    }
  }

  if (update) {
    const generated = generateEnums(pyEnums);
    const constantsPath = resolve(repoRoot, 'src/constants.js');
    const original = fs.readFileSync(constantsPath, 'utf8');
    const start = original.indexOf('// Auto-generated enumeration maps');
    const end = original.indexOf('export function', start);
    if (start === -1 || end === -1) {
      console.error('Could not locate enumeration section in constants.js');
      process.exit(1);
    }
    const updated = generated + '\n' + original.slice(end);
    fs.writeFileSync(constantsPath, updated);
    console.log('Updated src/constants.js');
  }

  if (!mismatches.length || update) {
    console.log('Enum/mask check passed');
  }
}

await main();
