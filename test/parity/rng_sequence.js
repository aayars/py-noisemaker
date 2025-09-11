import { setSeed, random, randomInt, choice } from '../../src/rng.js';

const seed = Number(process.argv[2]);
const count = Number(process.argv[3]);

setSeed(seed);
const rand = [];
for (let i = 0; i < count; i++) {
  rand.push(random());
}

setSeed(seed);
const randInt = [];
for (let i = 0; i < count; i++) {
  randInt.push(randomInt(0, 99));
}

setSeed(seed);
const randIntSwap = [];
for (let i = 0; i < count; i++) {
  randIntSwap.push(randomInt(99, 0));
}

setSeed(seed);
const arr = Array.from({ length: 10 }, (_, i) => i);
const choices = [];
for (let i = 0; i < count; i++) {
  choices.push(choice(arr));
}

console.log(JSON.stringify({ random: rand, randomInt: randInt, randomIntSwap: randIntSwap, choice: choices }));
