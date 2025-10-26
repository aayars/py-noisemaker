import assert from 'assert';
import { tokenize } from '../js/noisemaker/dsl/tokenizer.js';
import { parse } from '../js/noisemaker/dsl/parser.js';

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

// Named argument using equals
const astEq = parse(tokenize('rotate(angle=45)'));
assert.strictEqual(astEq.body.type, 'CallExpr');

// Python-style conditional expression
const astIf = parse(tokenize('1 if true else 2'));
assert.strictEqual(astIf.body.type, 'TernaryExpr');

// Unary expressions
const unary = parse(tokenize('-1'));
assert.strictEqual(unary.body.type, 'UnaryExpr');

// Unary after multiplication
const mulUnaryMinus = parse(tokenize('1 * -2'));
assert.strictEqual(mulUnaryMinus.body.type, 'BinaryExpr');
const mulUnaryPlus = parse(tokenize('1 * +2'));
assert.strictEqual(mulUnaryPlus.body.type, 'BinaryExpr');

// C-style comments should be ignored
const commentSource = `
// Leading single-line comment
{
    /* Multi-line
       comment */
    layers: [noise()] // Trailing single-line comment
}
`;
const commentAst = parse(tokenize(commentSource));
assert.strictEqual(commentAst.body.type, 'ObjectExpr');

// Unterminated multi-line comment should throw
error = null;
try {
  tokenize('/* unterminated comment');
} catch (e) {
  error = e;
}
assert.ok(error instanceof Error);
assert.ok(error.message.includes('Unterminated multi-line comment'));

console.log('parser tests passed');
