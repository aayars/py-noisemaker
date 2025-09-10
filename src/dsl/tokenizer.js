const PUNCT = new Set(['(', ')', '{', '}', '[', ']', ',', ':', '.', '+', '-', '*', '/']);

function isDigit(ch) {
  return ch >= '0' && ch <= '9';
}

function isHex(ch) {
  return isDigit(ch) || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
}

function isIdentStart(ch) {
  return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch === '_';
}

function isIdent(ch) {
  return isIdentStart(ch) || isDigit(ch);
}

export function tokenize(source) {
  const tokens = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    const start = i;

    // Number (with optional leading dot)
    if (
      isDigit(ch) ||
      (ch === '.' && isDigit(source[i + 1]))
    ) {
      let num = '';
      if (ch === '.') {
        num = '0';
      }
      while (isDigit(source[i])) {
        num += source[i++];
      }
      if (source[i] === '.') {
        num += source[i++];
        while (isDigit(source[i])) num += source[i++];
      }
      tokens.push({ type: 'number', value: parseFloat(num), raw: num, start, end: i });
      continue;
    }

    if (PUNCT.has(ch)) {
      tokens.push({ type: ch, value: ch, start, end: i + 1 });
      i++;
      continue;
    }

    if (ch === '#') {
      i++;
      let hex = '';
      while (isHex(source[i])) hex += source[i++];
      if (hex.length !== 3 && hex.length !== 6) {
        throw new Error(`Invalid color at ${start}`);
      }
      tokens.push({ type: 'color', value: `#${hex}`, start, end: i });
      continue;
    }

    if (ch === '"') {
      i++;
      let str = '';
      while (i < source.length && source[i] !== '"') {
        str += source[i++];
      }
      if (source[i] !== '"') {
        throw new Error('Unterminated string literal');
      }
      i++;
      tokens.push({ type: 'string', value: str, start, end: i });
      continue;
    }

    if (isIdentStart(ch)) {
      let id = '';
      while (isIdent(source[i])) id += source[i++];
      if (id === 'true' || id === 'false') {
        tokens.push({ type: 'boolean', value: id === 'true', start, end: i });
      } else {
        tokens.push({ type: 'identifier', value: id, start, end: i });
      }
      continue;
    }

    throw new Error(`Unexpected character '${ch}' at ${i}`);
  }
  return tokens;
}
