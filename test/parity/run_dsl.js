import { tokenize } from '../../js/noisemaker/dsl/tokenizer.js';
import { parse } from '../../js/noisemaker/dsl/parser.js';
import { evaluate } from '../../js/noisemaker/dsl/evaluator.js';
import { setSeed } from '../../js/noisemaker/rng.js';

const [, , op, b64, seedStr, countStr] = process.argv;
const source = Buffer.from(b64, 'base64').toString('utf8');
if (seedStr) {
  setSeed(parseInt(seedStr, 10));
}
const count = countStr ? parseInt(countStr, 10) : 1;

let out;
if (op === 'tokenize') {
  out = tokenize(source);
} else if (op === 'parse') {
  out = parse(tokenize(source));
} else if (op === 'evaluate') {
  const ast = parse(tokenize(source));
  const runOnce = () => {
    let result = evaluate(ast);
    if (typeof result === 'function') {
      result = result({});
    }
    return result;
  };
  if (count > 1) {
    const results = [];
    for (let i = 0; i < count; i++) {
      results.push(runOnce());
    }
    out = results;
  } else {
    out = runOnce();
  }
} else {
  throw new Error(`Unknown op ${op}`);
}
console.log(JSON.stringify(out));
