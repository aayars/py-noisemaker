import crypto from 'crypto';
import { setSeed, random, randomInt, choice } from '../../src/rng.js';
import { setSeed as setValueSeed, valueNoise } from '../../src/value.js';
import { basic, multires } from '../../src/generators.js';
import * as effects from '../../src/effects.js';
import { cloudPoints } from '../../src/points.js';
import { render } from '../../src/composer.js';
import PRESETS from '../../src/presets.js';

const SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
];

async function hashTensor(tensor) {
    const arr = await tensor.read();
    const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    return crypto.createHash('sha256').update(buf).digest('hex');
}

async function generatorHashes() {
    const basicHashes = {};
    const multiresHashes = {};
    for (const seed of SEEDS) {
        setSeed(seed);
        setValueSeed(seed);
        const basicTensor = await basic(2, [128, 128, 3], { hueRotation: 0 });
        basicHashes[seed] = await hashTensor(basicTensor);

        setSeed(seed);
        setValueSeed(seed);
        const multiresTensor = await multires(2, [128, 128, 3], {
            octaves: 2,
            hueRotation: 0,
            postEffects: [],
            finalEffects: [],
        });
        multiresHashes[seed] = await hashTensor(multiresTensor);
    }
    return { basic: basicHashes, multires: multiresHashes };
}

const EFFECTS = {
    adjust_hue: effects.adjustHueEffect,
    adjust_saturation: effects.saturation,
    adjust_brightness: effects.adjustBrightness,
    adjust_contrast: effects.adjustContrast,
    posterize: effects.posterize,
    blur: effects.blur,
    bloom: effects.bloom,
    vignette: effects.vignette,
    vaseline: effects.vaseline,
    shadow: effects.shadow,
    warp: effects.warp,
    ripple: effects.ripple,
    wobble: effects.wobble,
    reverb: effects.reverb,
    light_leak: effects.lightLeak,
    crt: effects.crt,
    scanline_error: effects.scanlineError,
    reindex: effects.reindex,
};

async function effectHashes() {
    const out = {};
    for (const [name, fn] of Object.entries(EFFECTS)) {
        out[name] = {};
        for (const seed of SEEDS) {
            setSeed(seed);
            setValueSeed(seed);
            const base = await basic(2, [128, 128, 3], { hueRotation: 0 });
            const effected = await fn(base, [128, 128, 3], 0, 1);
            out[name][seed] = await hashTensor(effected);
        }
    }
    return out;
}

function rngParity() {
    const COUNT = 10;
    const out = {};
    const arr = Array.from({ length: 10 }, (_, i) => i);
    for (const seed of SEEDS.slice(0, 3)) {
        setSeed(seed);
        const rand = [];
        for (let i = 0; i < COUNT; i++) {
            rand.push(random());
        }

        setSeed(seed);
        const randInt = [];
        for (let i = 0; i < COUNT; i++) {
            randInt.push(randomInt(0, 99));
        }

        setSeed(seed);
        const randIntSwap = [];
        for (let i = 0; i < COUNT; i++) {
            randIntSwap.push(randomInt(99, 0));
        }

        setSeed(seed);
        const choices = [];
        for (let i = 0; i < COUNT; i++) {
            choices.push(choice(arr));
        }

        out[seed] = {
            random: rand,
            randomInt: randInt,
            randomIntSwap: randIntSwap,
            choice: choices,
        };
    }
    return out;
}

function pointsParity() {
    const out = {};
    for (const seed of SEEDS.slice(0, 3)) {
        setSeed(seed);
        const [x, y] = cloudPoints(4);
        out[seed] = { x, y };
    }
    return out;
}

function valueParity() {
    const out = {};
    for (const seed of SEEDS.slice(0, 3)) {
        setSeed(seed);
        const arr = Array.from(valueNoise(64));
        out[seed] = arr;
    }
    return out;
}

async function composerParity() {
    const presetNames = ['basic', 'worms', 'voronoi'];
    const out = {};
    for (const name of presetNames) {
        setSeed(1);
        const presets = PRESETS();
        const result = await render(name, 1, { presets, width: 128, height: 128, debug: false });
        out[name] = { effects: result.effects, rng_calls: result.rngCalls };
    }
    return out;
}

const data = {
    generators: await generatorHashes(),
    effects: await effectHashes(),
    rng: rngParity(),
    points: pointsParity(),
    value: valueParity(),
    composer: await composerParity(),
};

console.log(JSON.stringify(data, null, 2));
