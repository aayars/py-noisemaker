const PRESET_KEYS = new Set(['layers', 'settings', 'generator', 'octaves', 'post', 'final', 'unique']);

function unexpected(token) {
  const val = token ? token.value : 'EOF';
  throw new Error(`Unexpected token: ${val}`);
}

export function parse(tokens, enforcePresetKeys = true) {
  let i = 0;

  function peek(offset = 0) {
    return tokens[i + offset];
  }

  function peekIs(type, offset = 0) {
    const t = peek(offset);
    return t && t.type === type;
  }

  function consume(type) {
    const t = peek();
    if (!t || (type && t.type !== type)) {
      unexpected(t);
    }
    i++;
    return t;
  }

  function match(type) {
    if (peekIs(type)) {
      i++;
      return true;
    }
    return false;
  }

  function parseProgram() {
    const t = peek();
    if (t && t.type === '{') {
      const body = parseObjectExpr(enforcePresetKeys);
      if (i !== tokens.length) unexpected(peek());
      return { type: 'Program', body };
    }
    const expr = parseExpression();
    if (i !== tokens.length) unexpected(peek());
    return { type: 'Program', body: expr };
  }

  function parseObjectExpr(enforceKeys = false) {
    consume('{');
    const properties = [];
    const seen = new Set();
    while (!peekIs('}')) {
      const keyTok = peek();
      if (!keyTok || (keyTok.type !== 'identifier' && keyTok.type !== 'string')) {
        unexpected(keyTok);
      }
      const key = consume(keyTok.type).value;
      if (seen.has(key)) {
        throw new Error(`Duplicate key: ${key}`);
      }
      if (enforceKeys && !PRESET_KEYS.has(key)) {
        throw new Error(`Unknown key: ${key}`);
      }
      seen.add(key);
      consume(':');
      const value = parseExpression();
      properties.push({ key, value });
      if (!match(',')) {
        break;
      }
    }
    consume('}');
    return { type: 'ObjectExpr', properties };
  }

  function parseArrayExpr() {
    consume('[');
    const elements = [];
    while (!peekIs(']')) {
      elements.push(parseExpression());
      if (!match(',')) {
        break;
      }
    }
    consume(']');
    return { type: 'ArrayExpr', elements };
  }

  function parseExpression() {
    const t = peek();
    if (!t) unexpected(t);
    if (t.type === 'number' || t.type === '(' || t.type === 'identifier' || t.type === 'boolean' || t.type === 'null') {
      return parseNumberExpr();
    }
    if (t.type === 'string') {
      consume('string');
      return { type: 'StringLiteral', value: t.value };
    }
    if (t.type === 'color') {
      consume('color');
      return { type: 'StringLiteral', value: t.value };
    }
    if (t.type === '[') {
      return parseArrayExpr();
    }
    if (t.type === '{') {
      return parseObjectExpr();
    }
    unexpected(t);
  }

  function parseCallChain() {
    let node = parseSingleCall();
    while (match('.')) {
      const next = parseSingleCall();
      next.input = node;
      node = next;
    }
    return node;
  }

  function parseSingleCall() {
    const callee = parseIdentifier();
    consume('(');
    const args = parseArgList(')');
    return { type: 'CallExpr', callee, args };
  }

  function parseArgList(endToken) {
    const positional = [];
    const named = {};
    let usingNamed = null;
    while (!peekIs(endToken)) {
      const arg = parseArg();
      if (arg.named) {
        if (usingNamed === false) {
          throw new Error('Cannot mix positional and named arguments');
        }
        usingNamed = true;
        if (Object.prototype.hasOwnProperty.call(named, arg.name)) {
          throw new Error(`Duplicate argument: ${arg.name}`);
        }
        named[arg.name] = arg.value;
      } else {
        if (usingNamed === true) {
          throw new Error('Cannot mix positional and named arguments');
        }
        usingNamed = false;
        positional.push(arg.value);
      }
      if (!match(',')) {
        break;
      }
    }
    consume(endToken);
    return usingNamed === true ? { named } : { positional };
  }

  function parseArg() {
    const t = peek();
    if (t.type === 'identifier' && peek(1)?.type === ':') {
      const name = consume('identifier').value;
      consume(':');
      const value = parseExpression();
      return { named: true, name, value };
    }
    return { value: parseExpression() };
  }

  function parseIdentifier() {
    const token = consume('identifier');
    return { type: 'Identifier', name: token.value };
  }

  function parseNumberExpr() {
    let node = parseAdd();
    if (match('?')) {
      const trueExpr = parseNumberExpr();
      consume(':');
      const falseExpr = parseNumberExpr();
      node = { type: 'TernaryExpr', test: node, consequent: trueExpr, alternate: falseExpr };
    }
    return node;
  }

  function parseAdd() {
    let node = parseMul();
    while (true) {
      const t = peek();
      if (t && (t.type === '+' || t.type === '-')) {
        consume(t.type);
        const right = parseMul();
        node = { type: 'BinaryExpr', operator: t.type, left: node, right };
      } else {
        break;
      }
    }
    return node;
  }

  function parseMul() {
    let node = parseNumberPrimary();
    while (true) {
      const t = peek();
      if (t && (t.type === '*' || t.type === '/')) {
        consume(t.type);
        const right = parseNumberPrimary();
        node = { type: 'BinaryExpr', operator: t.type, left: node, right };
      } else {
        break;
      }
    }
    return node;
  }

  function parseNumberPrimary() {
    const t = peek();
    if (t.type === 'number') {
      consume('number');
      return { type: 'NumberLiteral', value: t.value };
    }
    if (t.type === '(') {
      consume('(');
      const expr = parseNumberExpr();
      consume(')');
      return expr;
    }
    if (
      t.type === 'identifier' &&
      t.value === 'Math' &&
      peek(1)?.type === '.' &&
      peek(2)?.type === 'identifier' &&
      peek(2).value === 'PI'
    ) {
      consume('identifier');
      consume('.');
      consume('identifier');
      return { type: 'NumberLiteral', value: Math.PI };
    }
    if (t.type === 'identifier') {
      if (peek(1)?.type === '(') {
        return parseCallChain();
      }
      const id = parseIdentifier();
      if (match('.')) {
        let member = parseIdentifier();
        let node = { type: 'MemberExpr', object: id, property: member };
        while (match('.')) {
          member = parseIdentifier();
          node = { type: 'MemberExpr', object: node, property: member };
        }
        if (match('(')) {
          const args = parseArgList(')');
          node = { type: 'CallExpr', callee: node, args };
          while (match('.')) {
            const next = parseSingleCall();
            next.input = node;
            node = next;
          }
          return node;
        }
        return node;
      }
      return id;
    }
    if (t.type === 'boolean') {
      consume('boolean');
      return { type: 'NumberLiteral', value: t.value ? 1 : 0 };
    }
    if (t.type === 'null') {
      consume('null');
      return { type: 'NullLiteral', value: null };
    }
    unexpected(t);
  }

  return parseProgram();
}
