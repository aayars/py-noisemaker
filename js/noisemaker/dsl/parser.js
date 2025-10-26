const PRESET_KEYS = new Set(['layers', 'settings', 'generator', 'octaves', 'post', 'final', 'unique']);

// Improved error reporting so that diagnostics include location information.
// This greatly simplifies debugging of complex DSL programs like the preset
// collection where a single unexpected token can otherwise be very difficult
// to track down.  When a parse error occurs we now surface both the token's
// textual value and its line/column position within the source.
function unexpected(token) {
  const val = token ? `${token.value} (line ${token.line}, column ${token.column})` : 'EOF';
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

  // Entry point for all expressions.  The grammar supports binary
  // operations on not only numbers but also arrays and strings.  We
  // therefore parse using a precedence climbing approach starting at
  // addition/subtraction.
  function parseExpression() {
    return parseTernary();
  }

  function parseTernary() {
    let node = parseComparison();
    if (peekIs('identifier') && peek().value === 'if') {
      consume('identifier'); // 'if'
      const testExpr = parseTernary();
      const elseTok = consume('identifier');
      if (elseTok.value !== 'else') unexpected(elseTok);
      const falseExpr = parseTernary();
      return { type: 'TernaryExpr', test: testExpr, consequent: node, alternate: falseExpr };
    } else if (match('?')) {
      const trueExpr = parseTernary();
      consume(':');
      const falseExpr = parseTernary();
      return { type: 'TernaryExpr', test: node, consequent: trueExpr, alternate: falseExpr };
    }
    return node;
  }

  function parseComparison() {
    let node = parseAdd();
    while (true) {
      const t = peek();
      if (t && (t.type === '<' || t.type === '>')) {
        consume(t.type);
        const right = parseAdd();
        node = { type: 'BinaryExpr', operator: t.type, left: node, right };
      } else {
        break;
      }
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
    let node = parseUnary();
    while (true) {
      const t = peek();
      if (t && (t.type === '*' || t.type === '/')) {
        consume(t.type);
        // Allow unary expressions on the right-hand side of a multiply or
        // divide.  Using `parsePrimary` here meant constructs like `1 * -2`
        // or `1 * +2` would fail to parse, producing an "Unexpected token: +"
        // or "Unexpected token: -" error.  By delegating to `parseUnary` we
        // permit prefix operators in these positions.
        const right = parseUnary();
        node = { type: 'BinaryExpr', operator: t.type, left: node, right };
      } else {
        break;
      }
    }
    return node;
  }

  function parseUnary() {
    const t = peek();
    if (t && (t.type === '+' || t.type === '-')) {
      consume(t.type);
      const argument = parseUnary();
      return { type: 'UnaryExpr', operator: t.type, argument };
    }
    return parsePrimary();
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
    const next = peek(1);
    if (t.type === 'identifier' && (next?.type === ':' || next?.type === '=')) {
      const name = consume('identifier').value;
      consume(next.type);
      const value = parseExpression();
      return { named: true, name, value };
    }
    return { value: parseExpression() };
  }

  function parseIdentifier() {
    const token = consume('identifier');
    return { type: 'Identifier', name: token.value };
  }

  function parsePrimary() {
    const t = peek();
    if (t.type === 'number') {
      consume('number');
      return { type: 'NumberLiteral', value: t.value };
    }
    if (t.type === 'string') {
      consume('string');
      return { type: 'StringLiteral', value: t.value };
    }
    if (t.type === 'color') {
      consume('color');
      return { type: 'StringLiteral', value: t.value };
    }
    if (t.type === 'boolean') {
      consume('boolean');
      return { type: 'BooleanLiteral', value: t.value };
    }
    if (t.type === 'null') {
      consume('null');
      return { type: 'NullLiteral', value: null };
    }
    if (t.type === '[') {
      return parseArrayExpr();
    }
    if (t.type === '{') {
      return parseObjectExpr();
    }
    if (t.type === '(') {
      consume('(');
      const expr = parseExpression();
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
    unexpected(t);
  }

  return parseProgram();
}
