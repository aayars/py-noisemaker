import crypto from 'crypto';
import { setSeed } from '../../src/rng.js';
import { setSeed as setSimplexSeed } from '../../src/simplex.js';
import { basic, multires } from '../../src/generators.js';
import * as effects from '../../src/effects.js';

const SEEDS = [
    3626764237, 1654615998, 3255389356, 3823568514, 1806341205,
    173879092, 1112038970, 4146640122, 2195908194, 2087043557,
    1739178872, 3943786419, 3366389305, 3564191072, 1302718217,
    4156669319, 2046968324, 1537810351, 2505606783, 3829653368,
];

function hashTensor(tensor) {
    const arr = tensor.read();
    const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    return crypto.createHash('sha256').update(buf).digest('hex');
}

function generatorHashes() {
    const basicHashes = {};
    const multiresHashes = {};
    for (const seed of SEEDS) {
        setSeed(seed);
        setSimplexSeed(seed);
        const basicTensor = basic(2, [128, 128, 3], { hueRotation: 0, seed });
        basicHashes[seed] = hashTensor(basicTensor);

        setSeed(seed);
        setSimplexSeed(seed);
        const multiresTensor = multires(2, [128, 128, 3], {
            octaves: 2,
            hueRotation: 0,
            postEffects: [],
            finalEffects: [],
            seed,
        });
        multiresHashes[seed] = hashTensor(multiresTensor);
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
    reindex: effects.reindex,
};

function effectHashes() {
    const out = {};
    for (const [name, fn] of Object.entries(EFFECTS)) {
        out[name] = {};
        for (const seed of SEEDS) {
            setSeed(seed);
            setSimplexSeed(seed);
            const base = basic(2, [128, 128, 3], { hueRotation: 0, seed });
            const effected = fn(base, [128, 128, 3], 0, 1);
            out[name][seed] = hashTensor(effected);
        }
    }
    return out;
}

const data = {
    generators: generatorHashes(),
    effects: effectHashes(),
};

console.log(JSON.stringify(data, null, 2));
