import { defaultContext } from './builtins.js';

function evalNumber(node) {
  if (node.type === 'Number') return node.value;
  if (node.type === 'Binary') {
    const l = evalNumber(node.left);
    const r = evalNumber(node.right);
    switch (node.op) {
      case '+':
        return l + r;
      case '-':
        return l - r;
      case '*':
        return l * r;
      case '/':
        return l / r;
      default:
        throw new Error(`Unknown operator ${node.op}`);
    }
  }
  throw new Error(`Unsupported number node: ${node.type}`);
}

function evalArg(node, ctx) {
  switch (node.type) {
    case 'Number':
    case 'Binary':
      return evalNumber(node);
    case 'String':
      return node.value;
    case 'Boolean':
      return node.value;
    case 'Color':
      return node.value;
    case 'List':
      return node.value.map((v) => evalArg(v, ctx));
    case 'Dict':
      return Object.fromEntries(
        node.value.map(({ key, value }) => [key, evalArg(value, ctx)])
      );
    case 'Enum':
      return ctx.enums?.[node.object]?.[node.member];
    case 'SourceRef':
      return ctx.surfaces[node.value];
    case 'OutputRef':
      return { output: node.value };
    case 'Ident':
      return ctx.operations[node.name] ?? ctx.surfaces[node.name];
    default:
      throw new Error(`Unsupported argument type: ${node.type}`);
  }
}

function evalArgs(arglist, ctx) {
  if (arglist.named) {
    const out = {};
    for (const [k, v] of Object.entries(arglist.named)) {
      out[k] = evalArg(v, ctx);
    }
    return [out];
  }
  return arglist.positional.map((a) => evalArg(a, ctx));
}

function evalCall(call, input, ctx) {
  const fn = ctx.operations[call.name];
  const args = evalArgs(call.args, ctx);
  if (typeof fn === 'function') {
    if (input !== undefined && input !== null) {
      return fn(input, ...args);
    }
    return fn(...args);
  }
  return { op: call.name, args, input };
}

function evalChain(chain, ctx) {
  let value = evalCall(chain.expr, undefined, ctx);
  for (const c of chain.calls) {
    value = evalCall(c, value, ctx);
  }
  return value;
}

export function evaluate(ast, ctx = defaultContext) {
  const value = evalChain(ast.chain, ctx);
  if (ast.out) {
    return { output: ast.out, value };
  }
  return value;
}
