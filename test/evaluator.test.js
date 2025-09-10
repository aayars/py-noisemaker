import assert from 'assert';
import { tokenize } from '../src/dsl/tokenizer.js';
import { parse } from '../src/dsl/parser.js';
import { evaluate } from '../src/dsl/evaluator.js';
import { setSeed } from '../src/util.js';

setSeed(1);

// Effect chain should attach metadata
const ast = parse(tokenize('rotate(angle: 45).posterize(levels: 5)'));
const result = evaluate(ast);
assert.strictEqual(result.__effectName, 'posterize');
assert.deepStrictEqual(result.__params, { levels: 5 });
assert.strictEqual(result.input.__effectName, 'rotate');
assert.deepStrictEqual(result.input.__params, { angle: 45 });

// Builtins should evaluate
const coin = evaluate(parse(tokenize('coinFlip()')));
assert.strictEqual(typeof coin, 'boolean');

const member = evaluate(parse(tokenize('randomMember([1,2,3])')));
assert.ok([1, 2, 3].includes(member));

evaluate(parse(tokenize('stash("x", 42)')));
const stashed = evaluate(parse(tokenize('stash("x")')));
assert.strictEqual(stashed, 42);

console.log('evaluator tests passed');

