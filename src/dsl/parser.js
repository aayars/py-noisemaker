function unexpected(token) {
  const val = token ? token.value : 'EOF';
  throw new Error(`Unexpected token: ${val}`);
}

export function parse(tokens) {
  let i = 0;

  function peek(offset = 0) {
    return tokens[i + offset];
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
    const t = peek();
    if (t && t.type === type) {
      i++;
      return true;
    }
    return false;
  }

  function parseProgram() {
    const chain = parseChain();
    let out = null;
    if (match('.')) {
      const id = consume('identifier');
      if (id.value !== 'out') unexpected(id);
      consume('(');
      if (peek() && peek().type === 'identifier' && /^o[0-9]$/.test(peek().value)) {
        out = consume('identifier').value;
      }
      consume(')');
    }
    return { type: 'Program', chain, out };
  }

  function parseChain() {
    const expr = parseCall();
    const calls = [];
    while (match('.')) {
      calls.push(parseCall());
    }
    return { type: 'Chain', expr, calls };
  }

  function parseCall() {
    const name = consume('identifier').value;
    consume('(');
    const args = parseArgList(')');
    consume(')');
    return { type: 'Call', name, args };
  }

  function parseArgList(terminator) {
    const args = [];
    const named = {};
    let usingNamed = null;
    while (!match(terminator)) {
      const token = peek();
      if (!token) unexpected(token);
      if (token.type === terminator) break;
      const arg = parseArg();
      if (arg.type === 'Named') {
        if (usingNamed === false) {
          throw new Error('Cannot mix positional and named arguments');
        }
        usingNamed = true;
        named[arg.name] = arg.value;
      } else {
        if (usingNamed === true) {
          throw new Error('Cannot mix positional and named arguments');
        }
        usingNamed = false;
        args.push(arg);
      }
      if (match(',')) {
        // allow trailing comma
        if (peek() && peek().type === terminator) {
          match(terminator);
          break;
        }
        continue;
      } else {
        break;
      }
    }
    return usingNamed ? { named } : { positional: args };
  }

  function parseArg() {
    const t = peek();
    if (t.type === 'identifier' && peek(1)?.type === ':') {
      const name = consume('identifier').value;
      consume(':');
      const value = parseArg();
      return { type: 'Named', name, value };
    }
    return parseValue();
  }

  function parseValue() {
    const t = peek();
    if (!t) unexpected(t);
    if (
      t.type === 'number' ||
      t.type === '(' ||
      (t.type === 'identifier' && (t.value === 'Math'))
    ) {
      return parseNumberExpr();
    }
    if (t.type === 'string') {
      consume('string');
      return { type: 'String', value: t.value };
    }
    if (t.type === 'boolean') {
      consume('boolean');
      return { type: 'Boolean', value: t.value ? 1 : 0 };
    }
    if (t.type === 'color') {
      consume('color');
      return { type: 'Color', value: t.value };
    }
    if (t.type === '[') {
      consume('[');
      const list = parseArgList(']');
      consume(']');
      return { type: 'List', value: list.positional || [] };
    }
    if (t.type === '{') {
      consume('{');
      const entries = [];
      while (!match('}')) {
        const keyTok = peek();
        if (keyTok.type !== 'string' && keyTok.type !== 'identifier') {
          unexpected(keyTok);
        }
        const key = consume(keyTok.type).value;
        consume(':');
        const value = parseArg();
        entries.push({ key, value });
        if (match(',')) {
          if (peek()?.type === '}') {
            match('}');
            break;
          }
          continue;
        } else {
          break;
        }
      }
      return { type: 'Dict', value: entries };
    }
    if (t.type === 'identifier') {
      const id = consume('identifier').value;
      if (/^o[0-9]$/.test(id)) {
        return { type: 'OutputRef', value: id };
      }
      const sources = ['synth1', 'synth2', 'mixer', 'post1', 'post2', 'post3', 'final'];
      if (sources.includes(id)) {
        return { type: 'SourceRef', value: id };
      }
      if (match('.')) {
        const member = consume('identifier').value;
        return { type: 'Enum', object: id, member };
      }
      return { type: 'Ident', name: id };
    }
    unexpected(t);
  }

  function parseNumberExpr() {
    return parseAdd();
  }

  function parseAdd() {
    let node = parseMul();
    while (true) {
      const t = peek();
      if (t && (t.type === '+' || t.type === '-')) {
        consume(t.type);
        const right = parseMul();
        node = { type: 'Binary', op: t.type, left: node, right };
      } else {
        break;
      }
    }
    return node;
  }

  function parseMul() {
    let node = parsePrimary();
    while (true) {
      const t = peek();
      if (t && (t.type === '*' || t.type === '/')) {
        consume(t.type);
        const right = parsePrimary();
        node = { type: 'Binary', op: t.type, left: node, right };
      } else {
        break;
      }
    }
    return node;
  }

  function parsePrimary() {
    const t = peek();
    if (t.type === 'number') {
      consume('number');
      return { type: 'Number', value: t.value };
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
      return { type: 'Number', value: Math.PI };
    }
    unexpected(t);
  }

  return parseProgram();
}
