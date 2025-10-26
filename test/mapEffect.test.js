import assert from 'assert';
import { mapEffect } from '../js/noisemaker/presets.js';

const runaway = () => runaway;
assert.throws(() => mapEffect(runaway, {}), /Runaway dynamic preset function/);

const runawayChain = () => () => runawayChain();
assert.throws(() => mapEffect(runawayChain, {}), /Runaway dynamic preset function/);

console.log('mapEffect tests passed');
