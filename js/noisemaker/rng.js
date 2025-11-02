let _callCount = 0;
const TAU = Math.PI * 2;

function normalizeCount(shape) {
    if (shape === undefined || shape === null) {
        return 1;
    }

    if (Array.isArray(shape)) {
        if (shape.length === 0) {
            return 1;
        }
        let total = 1;
        for (const dim of shape) {
            const value = Math.trunc(dim);
            if (!Number.isFinite(value) || value <= 0) {
                return 0;
            }
            total *= value;
        }
        return total;
    }

    const value = Math.trunc(shape);
    if (!Number.isFinite(value) || value <= 0) {
        return 0;
    }
    return value;
}

function buildArray(count) {
    return count > 0 ? new Float32Array(count) : new Float32Array(0);
}

export function uniform(count, min = 0, max = 1) {
    if (count === undefined || count === null) {
        return min + (max - min) * random();
    }

    const total = normalizeCount(count);
    const out = buildArray(total);
    const span = max - min;
    for (let i = 0; i < total; i++) {
        out[i] = Math.fround(min + span * random());
    }
    return out;
}

export function normal(count, mean = 0, stddev = 1) {
    if (count === undefined || count === null) {
        let u1 = 0;
        do {
            u1 = random();
        } while (u1 <= 0);
        const u2 = random();
        const mag = Math.sqrt(-2 * Math.log(u1));
        const z0 = mag * Math.cos(TAU * u2);
        return Math.fround(mean + stddev * z0);
    }

    const total = normalizeCount(count);
    const out = buildArray(total);
    let index = 0;
    while (index < total) {
        let u1 = 0;
        do {
            u1 = random();
        } while (u1 <= 0);
        const u2 = random();
        const mag = Math.sqrt(-2 * Math.log(u1));
        const z0 = mag * Math.cos(TAU * u2);
        out[index] = Math.fround(mean + stddev * z0);
        index += 1;
        if (index < total) {
            const z1 = mag * Math.sin(TAU * u2);
            out[index] = Math.fround(mean + stddev * z1);
            index += 1;
        }
    }
    return out;
}

export class Random {
    /**
     * Creates a seeded instance of the Random class
     * @constructor Random
     * @param {number} seed - The seed
     */
    constructor(seed) {
        if (seed === undefined || seed === null) { seed = Date.now(); }
        this.state = seed >>> 0;
    }

    /**
     * Core mulberry32 random implementation returning [0,1).
     * Advances internal state each call.
     * @returns {number}
     */
    random() {
        _callCount++;
        let t = (this.state + 0x6D2B79F5) >>> 0;
        t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
        t ^= (t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0;
        this.state = t >>> 0;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    /**
     * Returns a random float in the specified range. If unspecified, between 0 and 1
     * @param {number} min - The min value - defaults to 0
     * @param {number} max - The max value - defaults to 1
     * @returns {number}
     */
    float(min = 0, max = 1) {
        return this.random() * (max - min) + min;
    }

    /**
     * Returns a random integer in the specified range (inclusive)
     * @param {number} min - The min value
     * @param {number} max - The max value
     * @returns {number}
     */
    randomInt(min, max) {
        if (max < min) {
            [min, max] = [max, min];
        }
        return Math.floor(this.random() * (max - min + 1)) + min;
    }

    /**
     * Returns a random item from an array
     * @param {Array} arr - The array
     * @returns item
     */
    choice(arr) {
        return arr[this.randomInt(0, arr.length - 1)];
    }

    floatFixed(min = 0, max = 1) {
        return parseFloat(this.float(min, max).toFixed(2));
    }

    int(min, max) {
        return Math.floor(this.float(min, max));
    }

    item(arr) {
        return this.choice(arr);
    }

    /**
     * Returns a random value from an object
     * @param {object} obj - The object
     * @returns value
     */
    object(obj) {
        const keys = Object.keys(obj);
        const index = Math.floor(this.random() * keys.length);
        return obj[keys[index]];
    }

    /**
     * Returns a random color in hexadecimal format
     * @returns {string}
     */
    hexColor() {
        const chars = 'abcdef0123456789'.split('');
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += this.item(chars);
        }
        return color;
    }

    /**
     * Returns a random emoji string
     * @returns {string}
     */
    emoji() {
        const emojiRange = [
            [0x1F600, 0x1F64F], // Emoticons
            [0x1F300, 0x1F5FF], // Miscellaneous Symbols and Pictographs
            [0x1F680, 0x1F6FF], // Transport and Map Symbols
            [0x1F700, 0x1F77F], // Alchemical Symbols
            [0x1F780, 0x1F7FF], // Geometric Shapes Extended
            [0x1F800, 0x1F8FF], // Supplemental Arrows-C
            [0x1F900, 0x1F9FF], // Supplemental Symbols and Pictographs
            [0x1FA00, 0x1FA6F]  // Chess Symbols
        ];

        const [rangeStart, rangeEnd] = this.item(emojiRange);
        const randomCodePoint = Math.floor(this.random() * (rangeEnd - rangeStart + 1)) + rangeStart;
        return String.fromCodePoint(randomCodePoint);
    }

    /**
     * Returns a custom ruleset for cellular automata
     * @returns {string}
     */
    ruleset() {
        const b = ['B'];
        const s = ['S'];

        for (let i = 0; i <= 8; i++) {
            if (this.random() > 0.75) b.push(i);
            if (this.random() > 0.75) s.push(i);
        }

        if (b.length === 1) b.push(3);
        if (s.length === 1) s.push(4);

        return `${b.join('')}${s.join('')}`;
    }
}

let _baseSeed = 0x12345678;
let _rng = new Random(_baseSeed);

export function setSeed(s) {
    _baseSeed = s >>> 0;
    _rng = new Random(_baseSeed);
}

export function getSeed() {
    return _rng.state >>> 0;
}

export function getBaseSeed() {
    return _baseSeed >>> 0;
}

export function random() {
    return _rng.random();
}

export function randomInt(min, max) {
    return _rng.randomInt(min, max);
}

export function choice(arr) {
    return _rng.choice(arr);
}

export { _rng as rng };

export function resetCallCount() {
    _callCount = 0;
}

export function getCallCount() {
    return _callCount;
}
