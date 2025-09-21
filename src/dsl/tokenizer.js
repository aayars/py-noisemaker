// Support ternary operator and equals sign for named arguments
const PUNCT = new Set(['(', ')', '{', '}', '[', ']', ',', ':', '.', '+', '-', '*', '/', '?', '=', '<', '>']);

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

function makeError(message, line, column) {
  const err = new Error(`${message} at line ${line} column ${column}`);
  err.line = line;
  err.column = column;
  return err;
}

/**
 * Convert source string to a list of tokens.
 *
 * Each token has shape: { type, value, line, column }
 *
 * @param {string} source
 * @returns {Array<object>}
 */
export function tokenize(source) {
  const tokens = [];
  let i = 0;
  let line = 1;
  let column = 1;

  const length = source.length;

  function peek(offset = 0) {
    return source[i + offset];
  }

  function advance() {
    const ch = source[i++];
    if (ch === '\n') {
      line++;
      column = 1;
    } else {
      column++;
    }
    return ch;
  }

  function skipWhitespace() {
    while (i < length) {
      const ch = peek();
      if (ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n') {
        advance();
        continue;
      }
      if (ch === '/' && peek(1) === '/') {
        advance();
        advance();
        while (i < length && peek() !== '\n') {
          advance();
        }
        continue;
      }
      if (ch === '/' && peek(1) === '*') {
        const commentLine = line;
        const commentColumn = column;
        advance();
        advance();
        let closed = false;
        while (i < length) {
          if (peek() === '*' && peek(1) === '/') {
            advance();
            advance();
            closed = true;
            break;
          }
          advance();
        }
        if (!closed) {
          throw makeError('Unterminated multi-line comment', commentLine, commentColumn);
        }
        continue;
      }
      break;
    }
  }

  while (i < length) {
    skipWhitespace();
    if (i >= length) break;

    const ch = peek();
    const tokenLine = line;
    const tokenColumn = column;

    // Number literal (support optional leading dot)
    if (isDigit(ch) || (ch === '.' && isDigit(peek(1)))) {
      let numStr = '';
      if (ch === '.') {
        numStr = '0';
        advance();
      }
      while (isDigit(peek())) {
        numStr += advance();
      }
      if (peek() === '.') {
        numStr += advance();
        while (isDigit(peek())) {
          numStr += advance();
        }
      }
      tokens.push({
        type: 'number',
        value: parseFloat(numStr),
        line: tokenLine,
        column: tokenColumn,
      });
      continue;
    }

    // String literal (single or double quotes)
    if (ch === '"' || ch === '\'') {
      const quote = advance();
      let str = '';
      while (i < length && peek() !== quote) {
        const c = advance();
        if (c === '\\') {
          if (i >= length) {
            throw makeError('Unterminated string literal', tokenLine, tokenColumn);
          }
          const esc = advance();
          const map = { n: '\n', r: '\r', t: '\t', '\\': '\\', '"': '"', '\'': '\'' };
          str += map[esc] !== undefined ? map[esc] : esc;
        } else if (c === '\n') {
          throw makeError('Unterminated string literal', tokenLine, tokenColumn);
        } else {
          str += c;
        }
      }
      if (peek() !== quote) {
        throw makeError('Unterminated string literal', tokenLine, tokenColumn);
      }
      advance();
      tokens.push({
        type: 'string',
        value: str,
        line: tokenLine,
        column: tokenColumn,
      });
      continue;
    }

    // Hex colour literal
    if (ch === '#') {
      advance();
      let hex = '';
      while (i < length && isHex(peek())) {
        hex += advance();
      }
      if (hex.length !== 3 && hex.length !== 6) {
        throw makeError('Invalid color', tokenLine, tokenColumn);
      }
      tokens.push({
        type: 'color',
        value: `#${hex}`,
        line: tokenLine,
        column: tokenColumn,
      });
      continue;
    }

    // Punctuation
    if (PUNCT.has(ch)) {
      advance();
      tokens.push({
        type: ch,
        value: ch,
        line: tokenLine,
        column: tokenColumn,
      });
      continue;
    }

    // Identifier / keywords
    if (isIdentStart(ch)) {
      let id = '';
      while (i < length && isIdent(peek())) {
        id += advance();
      }
      if (id === 'true' || id === 'false') {
        tokens.push({
          type: 'boolean',
          value: id === 'true',
          line: tokenLine,
          column: tokenColumn,
        });
      } else if (id === 'null') {
        tokens.push({
          type: 'null',
          value: null,
          line: tokenLine,
          column: tokenColumn,
        });
      } else {
        tokens.push({
          type: 'identifier',
          value: id,
          line: tokenLine,
          column: tokenColumn,
        });
      }
      continue;
    }

    // Unrecognised character
    throw makeError(`Unexpected character '${ch}'`, tokenLine, tokenColumn);
  }

  return tokens;
}

