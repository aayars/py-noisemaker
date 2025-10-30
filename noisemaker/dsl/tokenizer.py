# Support ternary operator and equals sign for named arguments
PUNCT = set("(){}[],.:+-*/?=<>")


class TokenizerError(Exception):
    """Custom exception for tokenizer errors with line and column tracking."""

    def __init__(self, message: str, line: int, column: int):
        super().__init__(f"{message} at line {line} column {column}")
        self.line = line
        self.column = column


def is_digit(ch):
    return ch is not None and "0" <= ch <= "9"


def is_hex(ch):
    return ch is not None and (is_digit(ch) or ("a" <= ch <= "f") or ("A" <= ch <= "F"))


def is_ident_start(ch):
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch == "_"


def is_ident(ch):
    return is_ident_start(ch) or is_digit(ch)


def make_error(message, line, column):
    return TokenizerError(message, line, column)


def tokenize(source):
    tokens = []
    i = 0
    line = 1
    column = 1
    length = len(source)

    def peek(offset=0):
        idx = i + offset
        return source[idx] if idx < length else None

    def advance():
        nonlocal i, line, column
        ch = source[i]
        i += 1
        if ch == "\n":
            line += 1
            column = 1
        else:
            column += 1
        return ch

    def skip_whitespace():
        nonlocal i
        while i < length:
            ch = peek()
            if ch in (" ", "\t", "\r", "\n"):
                advance()
                continue
            if ch == "/" and peek(1) == "/":
                advance()
                advance()
                while i < length and peek() != "\n":
                    advance()
                continue
            if ch == "/" and peek(1) == "*":
                comment_line = line
                comment_column = column
                advance()
                advance()
                closed = False
                while i < length:
                    if peek() == "*" and peek(1) == "/":
                        advance()
                        advance()
                        closed = True
                        break
                    advance()
                if not closed:
                    raise make_error("Unterminated multi-line comment", comment_line, comment_column)
                continue
            break

    while i < length:
        skip_whitespace()
        if i >= length:
            break
        ch = peek()
        token_line = line
        token_column = column
        if is_digit(ch) or (ch == "." and is_digit(peek(1))):
            num_str = ""
            if ch == ".":
                num_str = "0"
                advance()
            while is_digit(peek()):
                num_str += advance()
            if peek() == ".":
                num_str += advance()
                while is_digit(peek()):
                    num_str += advance()
            tokens.append({"type": "number", "value": float(num_str), "line": token_line, "column": token_column})
            continue
        if ch in ('"', "'"):
            quote = advance()
            s = ""
            while i < length and peek() != quote:
                c = advance()
                if c == "\\":
                    if i >= length:
                        raise make_error("Unterminated string literal", token_line, token_column)
                    esc = advance()
                    mapping = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
                    s += mapping.get(esc, esc)
                elif c == "\n":
                    raise make_error("Unterminated string literal", token_line, token_column)
                else:
                    s += c
            if peek() != quote:
                raise make_error("Unterminated string literal", token_line, token_column)
            advance()
            tokens.append({"type": "string", "value": s, "line": token_line, "column": token_column})
            continue
        if ch == "#":
            advance()
            hex_str = ""
            while i < length and is_hex(peek()):
                hex_str += advance()
            if len(hex_str) not in (3, 6):
                raise make_error("Invalid color", token_line, token_column)
            tokens.append({"type": "color", "value": f"#{hex_str}", "line": token_line, "column": token_column})
            continue
        if ch in PUNCT:
            advance()
            tokens.append({"type": ch, "value": ch, "line": token_line, "column": token_column})
            continue
        if is_ident_start(ch):
            ident = ""
            while i < length and is_ident(peek()):
                ident += advance()
            if ident in ("true", "false"):
                tokens.append({"type": "boolean", "value": ident == "true", "line": token_line, "column": token_column})
            elif ident == "null":
                tokens.append({"type": "null", "value": None, "line": token_line, "column": token_column})
            else:
                tokens.append({"type": "identifier", "value": ident, "line": token_line, "column": token_column})
            continue
        raise make_error(f"Unexpected character '{ch}'", token_line, token_column)
    return tokens
