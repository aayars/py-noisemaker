import assert from 'assert';
import { tokenize } from '../src/dsl/tokenizer.js';
import { parse } from '../src/dsl/parser.js';

// Basic parse
const ast = parse(tokenize('{ layers: [noise()] }'));
assert.strictEqual(ast.type, 'Program');
assert.strictEqual(ast.body.type, 'ObjectExpr');

// Unknown key should throw
let error = null;
try {
  parse(tokenize('{ foo: 1 }'));
} catch (e) {
  error = e;
}
assert.ok(error instanceof Error);
assert.ok(error.message.includes('Unknown key'));

// Duplicate key should throw
error = null;
try {
  parse(tokenize('{ layers: [], layers: [] }'));
} catch (e) {
  error = e;
}
assert.ok(error instanceof Error);
assert.ok(error.message.includes('Duplicate key'));

// Ternary parsing
let tern = parse(tokenize('coin_flip() ? 1 : 2'));
assert.strictEqual(tern.body.type, 'TernaryExpr');

// Null literal parsing
let n = parse(tokenize('null'));
assert.strictEqual(n.body.type, 'NullLiteral');

console.log('parser tests passed');
