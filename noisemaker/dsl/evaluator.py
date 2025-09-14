from .builtins import defaultContext
import enum


def _has_function(v):
    if callable(v):
        return True
    if isinstance(v, list):
        return any(_has_function(x) for x in v)
    if isinstance(v, dict):
        return any(_has_function(x) for x in v.values())
    return False


def _resolve_value(v, settings):
    if callable(v) and not isinstance(v, enum.EnumMeta):
        return v(settings)
    if isinstance(v, list):
        return [_resolve_value(x, settings) for x in v]
    if isinstance(v, dict):
        return {k: _resolve_value(x, settings) for k, x in v.items()}
    return v


def _apply_unary(val, op):
    if op == '+':
        return val if isinstance(val, list) else +val
    if op == '-':
        if isinstance(val, list):
            return [-x for x in val]
        return -val
    raise ValueError(f"Unknown operator {op}")


def _apply_binary(l, r, op):
    if op == '+':
        if isinstance(l, list) and isinstance(r, list):
            return l + r
        if isinstance(l, list) and isinstance(r, (int, float)):
            return [n + r for n in l]
        if isinstance(l, (int, float)) and isinstance(r, list):
            return [l + n for n in r]
        return l + r
    if op == '-':
        if isinstance(l, list) and isinstance(r, list):
            return [n - r[i] for i, n in enumerate(l)]
        if isinstance(l, list) and isinstance(r, (int, float)):
            return [n - r for n in l]
        if isinstance(l, (int, float)) and isinstance(r, list):
            return [l - n for n in r]
        return l - r
    if op == '*':
        if isinstance(l, list) and isinstance(r, list):
            return [n * r[i] for i, n in enumerate(l)]
        if isinstance(l, list) and isinstance(r, (int, float)):
            out = []
            for _ in range(int(r)):
                out.extend(l)
            return out
        if isinstance(l, (int, float)) and isinstance(r, list):
            out = []
            for _ in range(int(l)):
                out.extend(r)
            return out
        return l * r
    if op == '/':
        if isinstance(l, list) and isinstance(r, (int, float)):
            return [n / r for n in l]
        if isinstance(l, (int, float)) and isinstance(r, list):
            return [l / n for n in r]
        return l / r
    if op == '<':
        return l < r
    if op == '>':
        return l > r
    raise ValueError(f"Unknown operator {op}")

def _eval_node(node, ctx):
    t = node['type']
    if t == 'NumberLiteral':
        return node['value']
    if t == 'StringLiteral':
        return node['value']
    if t == 'BooleanLiteral':
        return node['value']
    if t == 'NullLiteral':
        return None
    if t == 'Identifier':
        operations = ctx.get('operations', {})
        surfaces = ctx.get('surfaces', {})
        enums = ctx.get('enums')
        name = node['name']
        if name in operations:
            return operations[name]
        if name in surfaces:
            return surfaces[name]
        if enums and hasattr(enums, name):
            return getattr(enums, name)
        return name
    if t == 'MemberExpr':
        obj = node['object']
        prop = node['property']['name']
        if obj['type'] == 'Identifier' and ctx.get('enumMethods'):
            methods = ctx['enumMethods'].get(obj['name'])
            if methods and prop in methods:
                return methods[prop]
        obj_val = _eval_node(obj, ctx)
        if isinstance(obj_val, dict):
            return obj_val.get(prop)
        return getattr(obj_val, prop, None)
    if t == 'ArrayExpr':
        return [_eval_node(el, ctx) for el in node['elements']]
    if t == 'ObjectExpr':
        out = {}
        for pair in node['properties']:
            out[pair['key']] = _eval_node(pair['value'], ctx)
        return out
    if t == 'UnaryExpr':
        val = _eval_node(node['argument'], ctx)
        op = node['operator']
        if _has_function(val):
            return lambda settings: _apply_unary(_resolve_value(val, settings), op)
        return _apply_unary(val, op)
    if t == 'BinaryExpr':
        l = _eval_node(node['left'], ctx)
        r = _eval_node(node['right'], ctx)
        op = node['operator']
        if _has_function(l) or _has_function(r):
            return lambda settings: _apply_binary(_resolve_value(l, settings), _resolve_value(r, settings), op)
        return _apply_binary(l, r, op)
    if t == 'TernaryExpr':
        test = _eval_node(node['test'], ctx)
        consequent = _eval_node(node['consequent'], ctx)
        alternate = _eval_node(node['alternate'], ctx)
        if _has_function(test) or _has_function(consequent) or _has_function(alternate):
            return lambda settings: (
                _resolve_value(consequent, settings)
                if _resolve_value(test, settings)
                else _resolve_value(alternate, settings)
            )
        return _eval_node(node['consequent'], ctx) if test else _eval_node(node['alternate'], ctx)
    if t == 'CallExpr':
        return _eval_call(node, ctx)
    raise ValueError(f"Unsupported node type: {t}")

def _eval_args(arglist, ctx):
    if 'named' in arglist:
        params = {}
        names = []
        for k, v in arglist['named'].items():
            params[k] = _eval_node(v, ctx)
            names.append(k)
        return {'args': [params], 'params': params, 'paramNames': names}
    arr = [_eval_node(a, ctx) for a in arglist.get('positional', [])]
    return {'args': arr, 'params': None, 'paramNames': None}

def _eval_call(node, ctx):
    input_val = _eval_node(node['input'], ctx) if node.get('input') else None
    evaled = _eval_args(node['args'], ctx)
    args = evaled['args']
    params = evaled['params']
    param_names = evaled['paramNames']
    callee = node['callee']
    if callee['type'] == 'Identifier':
        name = callee['name']
        fn = ctx.get('operations', {}).get(name)
        if callable(fn):
            if getattr(fn, "__thunk", False) or _has_function(args) or _has_function(input_val):
                return lambda settings: fn(*_resolve_value(args, settings))
            return fn(*args)
        out = {'op': name, 'args': args, 'input': input_val, '__effectName': name}
        if param_names:
            out['__paramNames'] = param_names
            out['__params'] = params
        return out
    fn = _eval_node(callee, ctx)
    if callable(fn):
        if getattr(fn, "__thunk", False) or _has_function(args) or _has_function(input_val):
            return lambda settings: fn(*_resolve_value(args, settings))
        return fn(*args)
    raise ValueError('Unsupported callee')

def evaluate(ast, ctx=defaultContext):
    return _eval_node(ast['body'], ctx)
