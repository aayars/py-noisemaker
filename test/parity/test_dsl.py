import pytest
from enum import Enum

from noisemaker.dsl.tokenizer import tokenize
from noisemaker.dsl.parser import parse
from noisemaker.dsl.evaluator import evaluate
from noisemaker import rng

from .utils import js_dsl


def _norm(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


SOURCES = [
    "{ layers: [noise()] }",
    "coin_flip() ? 1 : 2",
    "rotate(angle=45).posterize(levels=5)",
    "enum_range(1,3)",
    "mask_freq(ValueMask.square, 1)",
    "1 if true else 2",
    "ColorSpace.color_members()",
]


@pytest.mark.parametrize("src", SOURCES)
def test_tokenizer_parity(src):
    py_tokens = tokenize(src)
    js_tokens = js_dsl("tokenize", src)
    assert py_tokens == js_tokens


@pytest.mark.parametrize("src", SOURCES)
def test_parser_parity(src):
    py_ast = parse(tokenize(src))
    js_ast = js_dsl("parse", src)
    assert py_ast == js_ast


EVAL_CASES = [
    ("enum_range(1,3)", None),
    ("mask_freq(ValueMask.square, 1)", None),
    ("random()", 1),
    ("random_int(1,3)", 1),
    ("coin_flip()", 1),
    ("random_member([1,2,3])", 1),
    ("stash('x', 42)", None),
    ("DistanceMetric.absolute_members()", None),
    ("DistanceMetric.all()", None),
    ("ColorSpace.color_members()", None),
    ("ValueMask.procedural_members()", None),
    ("ValueMask.grid_members()", None),
    ("ValueMask.glyph_members()", None),
    ("ValueMask.nonprocedural_members()", None),
    ("ValueMask.rgb_members()", None),
    ("PointDistribution.circular_members()", None),
    ("PointDistribution.grid_members()", None),
    ("WormBehavior.all()", None),
    ("masks.mask_shape(ValueMask.square)", None),
    ("masks.square_masks()", None),
]


@pytest.mark.parametrize("src, seed", EVAL_CASES)
def test_evaluator_parity(src, seed):
    if seed is not None:
        rng.set_seed(seed)
    result = evaluate(parse(tokenize(src)))
    if callable(result):
        result = result({})
    py_val = _norm(result)
    js_val = _norm(js_dsl("evaluate", src, seed))
    assert py_val == js_val


SEQ_CASES = [
    ("random()", 1, 5),
    ("random_int(1,3)", 1, 5),
    ("coin_flip()", 1, 5),
    ("random_member([1,2,3])", 1, 5),
]


@pytest.mark.parametrize("src, seed, count", SEQ_CASES)
def test_evaluator_sequence_parity(src, seed, count):
    rng.set_seed(seed)
    ast = parse(tokenize(src))
    py_vals = []
    for _ in range(count):
        result = evaluate(ast)
        if callable(result):
            result = result({})
        py_vals.append(_norm(result))
    js_vals = [_norm(v) for v in js_dsl("evaluate", src, seed, count)]
    assert py_vals == js_vals
