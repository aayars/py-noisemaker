from .builtins import defaultContext

def _eval_node(node, ctx):
    t = node['type']
    if t == 'NumberLiteral':
        return node['value']
    if t == 'StringLiteral':
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
        obj_val = _eval_node(node['object'], ctx)
        prop = node['property']['name']
        return getattr(obj_val, prop, None)
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
    if t == 'TernaryExpr':
        test = _eval_node(node['test'], ctx)
        branch = node['consequent'] if test else node['alternate']
        return _eval_node(branch, ctx)
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
            return fn(*args)
        out = {'op': name, 'args': args, 'input': input_val}
        out['__effectName'] = name
        if param_names:
            out['__paramNames'] = param_names
            out['__params'] = params
        return out
    fn = _eval_node(callee, ctx)
    if callable(fn):
        return fn(*args)
    raise ValueError('Unsupported callee')

def evaluate(ast, ctx=defaultContext):
    return _eval_node(ast['body'], ctx)
