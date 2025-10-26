# Noisemaker Preset DSL Specification

This document describes a domain‑specific language (DSL) for declaring
Noisemaker presets in the JavaScript demo.  The DSL allows users to edit and
evaluate preset formulas inside a `<textarea>` without executing arbitrary
JavaScript.  Programs are parsed into an abstract syntax tree (AST) and
evaluated against a whitelist of operations and surfaces.

When behaviour is unspecified, refer to the Python implementation for
clarification.

## 1. Goals

* Enable interactive editing of presets in `demo.html` via a form
  `<textarea>`.
* Provide an evaluation environment that mirrors the current preset
  declarations while preventing arbitrary code execution.
* Offer clear diagnostics with source spans for syntax and runtime problems.
* Maintain feature parity with the Python preset DSL where feasible.

## 2. Execution Model

1. **Tokenize** – convert the source string to a stream of tokens.
2. **Parse** – build an AST according to the grammar below.
3. **Validate** – resolve identifiers and arguments, producing diagnostics for
   unknown symbols or out‑of‑range values.
4. **Evaluate** – execute the AST against the preset context, returning either
   a rendered surface or an error code.  Boolean literals evaluate to `1` and
   `0`.

The evaluator never invokes `eval` or `Function` and only dispatches to known
operations.  Runtime errors are surfaced through diagnostic `R001`.

## 3. Grammar

Extended EBNF describing the language:

```ebnf
Program        ::= Chain ('.' 'out' '(' OutputRef? ')' )?
Chain          ::= Expr ('.' Call)*
Expr           ::= Ident '(' ArgList? ')'
Call           ::= Ident '(' ArgList? ')'
ArgList        ::= Arg (',' Arg)* ','?
Arg            ::= NumberExpr | String | Boolean | Color
                 | Ident | Enum | OutputRef | SourceRef | Null
                 | List | Dict
NumberExpr     ::= Primary
                 | NumberExpr ( '+' | '-' | '*' | '/' ) NumberExpr
                 | NumberExpr '?' NumberExpr ':' NumberExpr
Primary        ::= Number | Boolean | Null | 'Math.PI'
                 | Ident | Enum | Call | MemberExpr
                 | '(' NumberExpr ')'
Call           ::= (Ident | MemberExpr) '(' ArgList? ')'
MemberExpr     ::= Ident '.' Ident
List           ::= '[' ArgList? ']'
Dict           ::= '{' (DictEntry (',' DictEntry)* ','?)? '}'
DictEntry      ::= (String | Ident) ':' Arg
Enum           ::= Ident '.' Ident
OutputRef      ::= 'o' Digit
SourceRef      ::= tbd
Ident          ::= Letter ( Letter | Digit | '_' )*
Number         ::= Digit+ ('.' Digit+)?
String         ::= '"' [^"\n]* '"'
Digit          ::= '0'…'9'
Letter         ::= 'A'…'Z' | 'a'…'z'
Boolean        ::= 'true' | 'false'
Null           ::= 'null'
Color          ::= '#' HexDigit HexDigit HexDigit
                 (HexDigit HexDigit HexDigit)?
HexDigit       ::= Digit | 'A'…'F' | 'a'…'f'
```

* Floats may omit the leading zero (e.g. `.5`).
* Trailing commas in lists, dictionaries and argument lists are allowed.
* `Enum` resolves to JavaScript enum objects exported from `constants.js`.
* Whitespace and comments are skipped between tokens.  The lexer recognises
  `//` single-line comments that run until the newline and `/* … */` block
  comments.  Block comments must be terminated before the end of the file;
  otherwise tokenisation fails with an "Unterminated multi-line comment"
  error.

## 4. Data Types

| Type        | Description |
|-------------|-------------|
| `Number`    | Supports inline arithmetic, ternary conditionals, function/method results, and the `Math.PI` constant. |
| `String`    | Double‑quoted strings without escapes. |
| `Boolean`   | Keywords `true` and `false`, coerced to `1` and `0`. |
| `Null`      | Keyword `null`. |
| `Color`     | Hex colours `#RGB` or `#RRGGBB`. |
| `OutputRef` | References a previously named output (`o1`–`o9`). |
| `SourceRef` | References built‑in surfaces. |
| `List`      | Ordered collection of `Arg` values. |
| `Dict`      | Mapping of string or identifier keys to `Arg` values. |
| `Enum`      | Enum access via `EnumType.Member` syntax. |

## 5. Functions and Arguments

Functions accept arguments **either** positionally **or** as named keywords;
mixing the two forms in a single call is a parser error.  Example:

```dsl
example(10, 0.1, 1)
example(foo: 10, bar: 0.1, baz: 1)
```

Identifiers must resolve to registered operations or surfaces.  Unknown
identifiers emit `S001` diagnostics.  Each operation defines its own parameter
validation, issuing `S002` when clamping out‑of‑range values.

### Naming Conventions

* Preset names use **kebab-case**.
* Preset setting keys use **snake_case**.
* Function wrappers and argument keys use **snake_case**.
* Enum wrapper members use **snake_case**.
* Underlying implementations:
  * Python enums and functions use `snake_case`.
  * JavaScript enums and functions use `camelCase`.

## 6. Diagnostics

| Code | Stage    | Severity | Message |
|------|----------|----------|---------|
| L001 | Lexer    | Error    | Unexpected character |
| L002 | Lexer    | Error    | Unterminated string literal |
| L003 | Lexer    | Error    | Unterminated multi-line comment |
| P001 | Parser   | Error    | Unexpected token |
| P002 | Parser   | Error    | Expected closing parenthesis |
| S001 | Semantic | Error    | Unknown identifier |
| S002 | Semantic | Warning  | Argument out of range |
| R001 | Runtime  | Error    | Runtime error |

Implementations may extend the list but existing codes and text must remain
stable.  Diagnostics carry source spans for editor highlighting.

## 7. Built‑In Variables and Enums

The preset context exposes read‑only surfaces and constants:

* Surfaces: tbd.
* Outputs: `o1`–`o9` created by the `.out(oN)` call suffix.
* Enumerations mirroring the Python `noisemaker.constants` module are
  available via the `EnumType.Member` syntax.

## 8. Integration with `demo.html`

1. Replace the read‑only `<pre id="presetFormula">` with a `<textarea>` for
   live editing.
2. On each change, tokenize, parse and validate the DSL.  Display diagnostics
   near the textarea and highlight erroneous spans.
3. If parsing and validation succeed, evaluate the AST to produce the preset
   object and render it to the canvas.
4. Show the evaluated preset structure in `#presetEvaluated` as today.
5. The evaluation happens inside the browser without invoking `eval` or
   executing user supplied JavaScript.

## 9. Security Considerations

* Only whitelisted operations and enums are accessible.
* The evaluator has no access to global scope or DOM APIs beyond rendering
  surfaces.
* Parsing failure or runtime errors do not crash the page; diagnostics are
  reported and evaluation aborted.

## 10. Future Work

* Additional data structures may be added as presets evolve.
* Consider syntax highlighting and autocomplete within the textarea.
* Validation rules should mirror those in the Python reference implementation.

