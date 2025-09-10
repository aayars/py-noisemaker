from .builtins import defaultContext

def _eval_node(node, ctx):
    t = node['type']
    if t == 'NumberLiteral':
        return node['value']
    if t == 'StringLiteral':
        return node['value']
    if t == 'Identifier':
        operations = ctx.get('operations', {})
        surfaces = ctx.get('surfaces', {})
        return operations.get(node['name']) or surfaces.get(node['name']) or node['name']
    if t == 'MemberExpr':
        obj = node['object']['name']
        prop = node['property']['name']
        enums = ctx.get('enums')
        enum_obj = getattr(enums, obj, None) if enums else None
        return getattr(enum_obj, prop, None) if enum_obj else None
    if t == 'ArrayExpr':
        return [_eval_node(el, ctx) for el in node['elements']]
    if t == 'ObjectExpr':
        out = {}
        for pair in node['properties']:
            out[pair['key']] = _eval_node(pair['value'], ctx)
        return out
    if t == 'BinaryExpr':
        l = _eval_node(node['left'], ctx)
        r = _eval_node(node['right'], ctx)
        op = node['operator']
        if op == '+':
            return l + r
        if op == '-':
            return l - r
        if op == '*':
            return l * r
        if op == '/':
            return l / r
        raise ValueError(f"Unknown operator {op}")
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
    name = node['callee']['name']
    evaled = _eval_args(node['args'], ctx)
    args = evaled['args']
    params = evaled['params']
    param_names = evaled['paramNames']
    fn = ctx.get('operations', {}).get(name)
    if callable(fn):
        return fn(*args)
    out = {'op': name, 'args': args, 'input': input_val}
    out['__effectName'] = name
    if param_names:
        out['__paramNames'] = param_names
        out['__params'] = params
    return out

def evaluate(ast, ctx=defaultContext):
    return _eval_node(ast['body'], ctx)
