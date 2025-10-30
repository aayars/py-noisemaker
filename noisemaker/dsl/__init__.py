from .builtins import defaultContext
from .evaluator import evaluate
from .parser import parse
from .tokenizer import tokenize


def parse_preset_dsl(source, context=defaultContext):
    tokens = tokenize(source)
    ast = parse(tokens, enforce_preset_keys=False)
    return evaluate(ast, context)


__all__ = [
    "tokenize",
    "parse",
    "evaluate",
    "parse_preset_dsl",
    "defaultContext",
]
