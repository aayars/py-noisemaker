from .tokenizer import tokenize
from .parser import parse
from .evaluator import evaluate
from .builtins import defaultContext


def parse_preset_dsl(source, context=defaultContext):
    tokens = tokenize(source)
    ast = parse(tokens)
    return evaluate(ast, context)

__all__ = [
    "tokenize",
    "parse",
    "evaluate",
    "parse_preset_dsl",
    "defaultContext",
]
