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
        let t = this.state += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        this.state = t >>> 0;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
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
     * Returns a random float in the specified range with two digits after the decimal point.
     * If range is unspecified, the value is between 0 and 1.
     * @param {number} min - The min value - defaults to 0
     * @param {number} max - The max value - defaults to 1
     * @returns {number} A floating point number with two decimal places
     */
    floatFixed(min = 0, max = 1) {
        return parseFloat(this.float(min, max).toFixed(2));
    }

    /**
     * Returns a random integer in the specified range
     * @param {number} min - The min value
     * @param {number} max - The max value (exclusive)
     * @returns {number}
     */
    int(min, max) {
        return Math.floor(this.float(min, max));
    }

    /**
     * Returns a random item from an array
     * @param {Array} arr - The array
     * @returns item
     */
    item(arr) {
        const index = Math.floor(this.random() * arr.length);
        return arr[index];
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

let _rng = new Random(0x12345678);

export function setSeed(s) {
    _rng = new Random(s);
}

export function getSeed() {
    return _rng.state >>> 0;
}

export function random() {
    return _rng.random();
}

export { _rng as rng };
