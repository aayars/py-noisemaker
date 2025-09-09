import assert from "assert";
import { register, EFFECTS, list } from "../src/effectsRegistry.js";

function valid(tensor, shape, time, speed, gain = 1, bias = 0) {
  return tensor;
}

register("valid", valid, { gain: 1, bias: 0 });
assert.strictEqual(EFFECTS.valid.func, valid);
assert.strictEqual(EFFECTS.valid.gain, 1);
assert.strictEqual(EFFECTS.valid.bias, 0);

function missingSpeed(tensor, shape, time) {}
assert.throws(() => register("missingSpeed", missingSpeed, {}));

function wrongName(foo, shape, time, speed) {}
assert.throws(() => register("wrongName", wrongName, {}));

function badDefaultName(tensor, shape, time, speed, gain = 1) {}
assert.throws(() => register("badDefaultName", badDefaultName, { bias: 0 }));

function badDefaultCount(tensor, shape, time, speed, gain, bias) {}
assert.throws(() => register("badDefaultCount", badDefaultCount, { gain: 1 }));

assert.ok(list().includes("valid"));
assert.ok(EFFECTS.list().includes("valid"));

console.log("effectsRegistry tests passed");
