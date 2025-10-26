import pytest
from noisemaker.dsl.tokenizer import tokenize
from noisemaker.dsl.parser import parse
from noisemaker.dsl.evaluator import evaluate
from noisemaker import rng

rng.set_seed(1)


def eval_call(source):
    result = evaluate(parse(tokenize(source)))
    return result({}) if callable(result) else result

def test_effect_chain_metadata():
    ast = parse(tokenize('rotate(angle: 45).posterize(levels: 5)'))
    result = evaluate(ast)
    assert result['__effectName'] == 'posterize'
    assert result['__params'] == {'levels': 5}
    assert result['input']['__effectName'] == 'rotate'
    assert result['input']['__params'] == {'angle': 45}


def test_effect_chain_metadata_equals():
    ast = parse(tokenize('rotate(angle=45).posterize(levels=5)'))
    result = evaluate(ast)
    assert result['__effectName'] == 'posterize'
    assert result['input']['__params'] == {'angle': 45}

def test_builtins():
    rng.set_seed(1)
    coin = eval_call('coin_flip()')
    assert isinstance(coin, bool)

    rng.set_seed(1)
    member = eval_call('random_member([1,2,3])')
    assert member in [1, 2, 3]

    eval_call('stash("x", 42)')
    stashed = eval_call('stash("x")')
    assert stashed == 42

    rng_range = eval_call('enum_range(1,3)')
    assert rng_range == [1, 2, 3]

    rng.set_seed(1)
    rnd = eval_call('random()')
    assert rnd == pytest.approx(0.6270739405881613)

    rng.set_seed(1)
    rnd_int = eval_call('random_int(1,3)')
    assert rnd_int == 2

def test_errors():
    with pytest.raises(Exception) as e:
        eval_call('coin_flip(1)')
    assert 'takes no arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        eval_call('enum_range(1)')
    assert 'requires exactly 2 arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        eval_call('enum_range("a",3)')
    assert 'requires numeric arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        eval_call('random_member()')
    assert 'requires at least one iterable argument' in str(e.value)

    with pytest.raises(Exception) as e:
        eval_call('stash(1)')
    assert 'key must be a string' in str(e.value)

    with pytest.raises(Exception) as e:
        eval_call('stash("x",1,2)')
    assert 'expects 1 or 2 arguments' in str(e.value)


def test_numeric_and_random_and_null():
    rng.set_seed(1)
    val = eval_call('random_int(1,3) + ValueDistribution.ones.value')
    assert val == 7

    rng.set_seed(1)
    tern = eval_call('coin_flip() ? 1 : 2')
    assert tern == 1

    rng.set_seed(1)
    tern_py = eval_call('1 if coin_flip() else 2')
    assert tern_py == 1

    rng.set_seed(1)
    member = eval_call('random_member(ColorSpace.color_members())')
    from noisemaker.constants import ColorSpace
    assert member in ColorSpace.color_members()

    null_val = eval_call('null')
    assert null_val is None


def test_unary_expression():
    rng.set_seed(1)
    val = eval_call('-1 * +2')
    assert val == -2
