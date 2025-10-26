import { tokenize } from './tokenizer.js';
import { parse } from './parser.js';
import { evaluate } from './evaluator.js';
import { defaultContext } from './builtins.js';

/**
 * Parse and evaluate a Preset DSL source string.
 *
 * This helper performs the three stages described in PRESET_DSL_SPEC.md:
 * tokenisation, parsing and evaluation.  The evaluation environment is
 * constrained to the whitelisted operations exposed by `builtins.js`.
 *
 * @param {string} source - DSL program source
 * @param {object} [context] - Optional evaluation context
 * @returns {*} result of evaluating the program
 */
export function parsePresetDSL(source, context = defaultContext) {
  const tokens = tokenize(source);
  const ast = parse(tokens, false);
  return evaluate(ast, context);
}
