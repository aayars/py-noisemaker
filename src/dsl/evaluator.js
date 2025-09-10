import { defaultContext } from './builtins.js';

function evalNode(node, ctx) {
  switch (node.type) {
    case 'NumberLiteral':
      return node.value;
    case 'StringLiteral':
      return node.value;
    case 'NullLiteral':
      return null;
    case 'Identifier': {
      const name = node.name;
      if (ctx.operations[name]) return ctx.operations[name];
      if (ctx.surfaces[name]) return ctx.surfaces[name];
      if (ctx.enums && ctx.enums[name]) return ctx.enums[name];
      return name;
    }
    case 'MemberExpr': {
      const objVal = evalNode(node.object, ctx);
      const prop = node.property.name;
      return objVal?.[prop];
    }
    case 'ArrayExpr':
      return node.elements.map((el) => evalNode(el, ctx));
    case 'ObjectExpr': {
      const out = {};
      for (const { key, value } of node.properties) {
        out[key] = evalNode(value, ctx);
      }
      return out;
    }
    case 'BinaryExpr': {
      const l = evalNode(node.left, ctx);
      const r = evalNode(node.right, ctx);
      switch (node.operator) {
        case '+':
          return l + r;
        case '-':
          return l - r;
        case '*':
          return l * r;
        case '/':
          return l / r;
        default:
          throw new Error(`Unknown operator ${node.operator}`);
      }
    }
    case 'TernaryExpr':
      return evalNode(evalNode(node.test, ctx) ? node.consequent : node.alternate, ctx);
    case 'CallExpr':
      return evalCall(node, ctx);
    default:
      throw new Error(`Unsupported node type: ${node.type}`);
  }
}

function evalArgs(arglist, ctx) {
  if (arglist.named) {
    const params = {};
    const names = [];
    for (const [k, v] of Object.entries(arglist.named)) {
      params[k] = evalNode(v, ctx);
      names.push(k);
    }
    return { args: [params], params, paramNames: names };
  }
  const arr = arglist.positional.map((a) => evalNode(a, ctx));
  return { args: arr, params: null, paramNames: null };
}

function evalCall(node, ctx) {
  const input = node.input ? evalNode(node.input, ctx) : undefined;
  const { args, params, paramNames } = evalArgs(node.args, ctx);
  const callee = node.callee;
  if (callee.type === 'Identifier') {
    const name = callee.name;
    const fn = ctx.operations[name];
    if (typeof fn === 'function') {
      return fn(...args);
    }
    const out = { op: name, args, input };
    out.__effectName = name;
    if (paramNames) {
      out.__paramNames = paramNames;
      out.__params = params;
    }
    return out;
  }
  const fn = evalNode(callee, ctx);
  if (typeof fn === 'function') {
    return fn(...args);
  }
  throw new Error('Unsupported callee');
}

export function evaluate(ast, ctx = defaultContext) {
  return evalNode(ast.body, ctx);
}

