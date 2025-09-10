import assert from 'assert';
import { tokenize } from '../src/dsl/tokenizer.js';
import { parse } from '../src/dsl/parser.js';
import { evaluate } from '../src/dsl/evaluator.js';
import { setSeed } from '../src/util.js';
import { ValueDistribution } from '../src/constants.js';

// Effect chain should attach metadata
const ast = parse(tokenize('rotate(angle: 45).posterize(levels: 5)'));
const result = evaluate(ast);
assert.strictEqual(result.__effectName, 'posterize');
assert.deepStrictEqual(result.__params, { levels: 5 });
assert.strictEqual(result.input.__effectName, 'rotate');
assert.deepStrictEqual(result.input.__params, { angle: 45 });

// Builtins should evaluate
setSeed(1);

let coin = evaluate(parse(tokenize('coin_flip()')));
assert.strictEqual(typeof coin, 'boolean');

setSeed(1);
let member = evaluate(parse(tokenize('random_member([1,2,3])')));
assert.ok([1, 2, 3].includes(member));

evaluate(parse(tokenize('stash("x", 42)')));
let stashed = evaluate(parse(tokenize('stash("x")')));
assert.strictEqual(stashed, 42);

let range = evaluate(parse(tokenize('enum_range(1,3)')));
assert.deepStrictEqual(range, [1, 2, 3]);

setSeed(1);
let rnd = evaluate(parse(tokenize('random()')));
assert.ok(rnd >= 0 && rnd < 1);

setSeed(1);
let rndInt = evaluate(parse(tokenize('random_int(1,3)')));
assert.ok([1, 2, 3].includes(rndInt));

// Arithmetic with calls and member expressions
setSeed(1);
let sum = evaluate(parse(tokenize('random_int(1,3) + ValueDistribution.ones')));
assert.strictEqual(sum, 7);

// Ternary operations
setSeed(1);
let tern = evaluate(parse(tokenize('coin_flip() ? 1 : 2')));
assert.strictEqual(tern, 2);

// Random enumeration selection
setSeed(1);
let enumPick = evaluate(parse(tokenize('random_member([ValueDistribution.ones, ValueDistribution.mids, ValueDistribution.zeros])')));
assert.ok([ValueDistribution.ones, ValueDistribution.mids, ValueDistribution.zeros].includes(enumPick));

// Null literal
let nullVal = evaluate(parse(tokenize('null')));
assert.strictEqual(nullVal, null);

assert.throws(() => evaluate(parse(tokenize('coin_flip(1)'))), /takes no arguments/);
assert.throws(
  () => evaluate(parse(tokenize('enum_range(1)'))),
  /requires exactly 2 arguments/
);
assert.throws(
  () => evaluate(parse(tokenize('enum_range("a",3)'))),
  /requires numeric arguments/
);
assert.throws(
  () => evaluate(parse(tokenize('random_member()'))),
  /requires at least one iterable argument/
);
assert.throws(
  () => evaluate(parse(tokenize('stash(1)'))),
  /key must be a string/
);
assert.throws(
  () => evaluate(parse(tokenize('stash("x",1,2)'))),
  /expects 1 or 2 arguments/
);

console.log('evaluator tests passed');
