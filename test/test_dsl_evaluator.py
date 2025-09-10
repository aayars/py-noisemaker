import pytest
from noisemaker.dsl.tokenizer import tokenize
from noisemaker.dsl.parser import parse
from noisemaker.dsl.evaluator import evaluate
from noisemaker import rng

rng.set_seed(1)

def test_effect_chain_metadata():
    ast = parse(tokenize('rotate(angle: 45).posterize(levels: 5)'))
    result = evaluate(ast)
    assert result['__effectName'] == 'posterize'
    assert result['__params'] == {'levels': 5}
    assert result['input']['__effectName'] == 'rotate'
    assert result['input']['__params'] == {'angle': 45}

def test_builtins():
    rng.set_seed(1)
    coin = evaluate(parse(tokenize('coin_flip()')))
    assert isinstance(coin, bool)

    rng.set_seed(1)
    member = evaluate(parse(tokenize('random_member([1,2,3])')))
    assert member in [1, 2, 3]

    evaluate(parse(tokenize('stash("x", 42)')))
    stashed = evaluate(parse(tokenize('stash("x")')))
    assert stashed == 42

    rng_range = evaluate(parse(tokenize('enum_range(1,3)')))
    assert rng_range == [1, 2, 3]

    rng.set_seed(1)
    rnd = evaluate(parse(tokenize('random()')))
    assert rnd == pytest.approx(0.6270739405881613)

    rng.set_seed(1)
    rnd_int = evaluate(parse(tokenize('random_int(1,3)')))
    assert rnd_int == 2

def test_errors():
    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('coin_flip(1)')))
    assert 'takes no arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('enum_range(1)')))
    assert 'requires exactly 2 arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('enum_range("a",3)')))
    assert 'requires numeric arguments' in str(e.value)

    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('random_member()')))
    assert 'requires at least one iterable argument' in str(e.value)

    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('stash(1)')))
    assert 'key must be a string' in str(e.value)

    with pytest.raises(Exception) as e:
        evaluate(parse(tokenize('stash("x",1,2)')))
    assert 'expects 1 or 2 arguments' in str(e.value)
