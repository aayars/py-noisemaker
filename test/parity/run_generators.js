import { setSeed, resetCallCount, getCallCount } from '../../src/rng.js';
import { setSeed as setValueSeed } from '../../src/value.js';
import { basic, multires } from '../../src/generators.js';

const [,, name, seedStr, optionsB64] = process.argv;
const seed = parseInt(seedStr, 10);
const options = optionsB64
  ? JSON.parse(Buffer.from(optionsB64, 'base64').toString('utf8'))
  : {};
const requestedShape = Array.isArray(options.shape)
  ? options.shape.map((v) => parseInt(v, 10))
  : null;
const shape = requestedShape && requestedShape.length === 3
  ? requestedShape
  : [128, 128, 3];
const generatorOptions = { ...options };
delete generatorOptions.shape;
setSeed(seed);
setValueSeed(seed);
resetCallCount();
let tensor;
if (name === 'basic') {
  tensor = await basic(2, shape, generatorOptions);
} else if (name === 'multires') {
  const merged = {
    octaves: 2,
    postEffects: [],
    finalEffects: [],
    ...generatorOptions,
  };
  const supersample =
    merged.withSupersample ?? merged.with_supersample ?? false;
  if (
    supersample &&
    merged.hueRotation === undefined &&
    merged.hue_rotation === undefined
  ) {
    merged.hueRotation = 0;
  }
  tensor = await multires(2, shape, merged);
} else {
  throw new Error(`Unknown generator ${name}`);
}
const arr = await tensor.read();
const buf = Buffer.from(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
console.log(
  JSON.stringify({
    tensor: buf.toString('base64'),
    callCount: getCallCount(),
    shape: Array.isArray(tensor.shape) ? tensor.shape : Array.from(tensor.shape ?? []),
  }),
);
