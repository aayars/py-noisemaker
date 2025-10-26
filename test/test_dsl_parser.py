import pytest
from noisemaker.dsl.tokenizer import tokenize
from noisemaker.dsl.parser import parse

def test_basic_parse():
    ast = parse(tokenize('{ layers: [noise()] }'))
    assert ast['type'] == 'Program'
    assert ast['body']['type'] == 'ObjectExpr'

def test_unknown_key():
    with pytest.raises(Exception) as e:
        parse(tokenize('{ foo: 1 }'))
    assert 'Unknown key' in str(e.value)

def test_duplicate_key():
    with pytest.raises(Exception) as e:
        parse(tokenize('{ layers: [], layers: [] }'))
    assert 'Duplicate key' in str(e.value)


def test_ternary_parse():
    ast = parse(tokenize('coin_flip() ? 1 : 2'))
    assert ast['body']['type'] == 'TernaryExpr'


def test_named_arg_equals():
    ast = parse(tokenize('rotate(angle=45)'))
    call = ast['body']
    assert call['type'] == 'CallExpr'
    assert call['args']['named']['angle']['type'] == 'NumberLiteral'


def test_python_style_conditional_parse():
    ast = parse(tokenize('1 if true else 2'))
    assert ast['body']['type'] == 'TernaryExpr'


def test_null_parse():
    ast = parse(tokenize('null'))
    assert ast['body']['type'] == 'NullLiteral'


def test_c_style_comments():
    source = '''
    // Leading single-line comment
    {
        /* Multi-line
           comment */
        layers: [noise()] // Trailing single-line comment
    }
    '''
    ast = parse(tokenize(source))
    assert ast['body']['type'] == 'ObjectExpr'


def test_unterminated_multiline_comment():
    with pytest.raises(Exception) as e:
        tokenize('/* unterminated comment')
    assert 'Unterminated multi-line comment' in str(e.value)
