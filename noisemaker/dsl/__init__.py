from __future__ import annotations

from typing import Any

from .builtins import defaultContext
from .evaluator import evaluate
from .parser import parse
from .tokenizer import tokenize


def parse_preset_dsl(source, context=defaultContext) -> dict[str, Any]:
    tokens = tokenize(source)
    ast = parse(tokens, enforce_preset_keys=False)
    result = evaluate(ast, context)
    assert isinstance(result, dict)
    return result


__all__ = [
    "tokenize",
    "parse",
    "evaluate",
    "parse_preset_dsl",
    "defaultContext",
]
