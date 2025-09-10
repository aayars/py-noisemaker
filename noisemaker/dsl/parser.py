PRESET_KEYS = {"layers", "settings", "generator", "octaves", "post", "final"}

def unexpected(token):
    val = token['value'] if token else 'EOF'
    raise ValueError(f"Unexpected token: {val}")

def parse(tokens):
    i = 0
    def peek(offset=0):
        idx = i + offset
        return tokens[idx] if idx < len(tokens) else None
    def peek_is(type_, offset=0):
        t = peek(offset)
        return t and t['type'] == type_
    def consume(type_=None):
        nonlocal i
        t = peek()
        if not t or (type_ and t['type'] != type_):
            unexpected(t)
        i += 1
        return t
    def match(type_):
        nonlocal i
        if peek_is(type_):
            i += 1
            return True
        return False
    def parseProgram():
        t = peek()
        if t and t['type'] == '{':
            body = parseObjectExpr(True)
            if i != len(tokens):
                unexpected(peek())
            return {'type': 'Program', 'body': body}
        expr = parseExpression()
        if i != len(tokens):
            unexpected(peek())
        return {'type': 'Program', 'body': expr}
    def parseObjectExpr(enforceKeys=False):
        consume('{')
        properties = []
        seen = set()
        while not peek_is('}'):
            keyTok = peek()
            if not keyTok or keyTok['type'] not in ('identifier', 'string'):
                unexpected(keyTok)
            key = consume(keyTok['type'])['value']
            if key in seen:
                raise ValueError(f"Duplicate key: {key}")
            if enforceKeys and key not in PRESET_KEYS:
                raise ValueError(f"Unknown key: {key}")
            seen.add(key)
            consume(':')
            value = parseExpression()
            properties.append({'key': key, 'value': value})
            if not match(','):
                break
        consume('}')
        return {'type': 'ObjectExpr', 'properties': properties}
    def parseArrayExpr():
        consume('[')
        elements = []
        while not peek_is(']'):
            elements.append(parseExpression())
            if not match(','):
                break
        consume(']')
        return {'type': 'ArrayExpr', 'elements': elements}
    def parseExpression():
        t = peek()
        if not t:
            unexpected(t)
        if t['type'] in ('number', '(', 'identifier', 'boolean', 'null'):
            return parseNumberExpr()
        if t['type'] == 'string':
            consume('string')
            return {'type': 'StringLiteral', 'value': t['value']}
        if t['type'] == 'color':
            consume('color')
            return {'type': 'StringLiteral', 'value': t['value']}
        if t['type'] == '[':
            return parseArrayExpr()
        if t['type'] == '{':
            return parseObjectExpr()
        unexpected(t)
    def parseCallChain():
        node = parseSingleCall()
        while match('.'):
            next_node = parseSingleCall()
            next_node['input'] = node
            node = next_node
        return node
    def parseSingleCall():
        callee = parseIdentifier()
        consume('(')
        args = parseArgList(')')
        return {'type': 'CallExpr', 'callee': callee, 'args': args}
    def parseArgList(endToken):
        positional = []
        named = {}
        usingNamed = None
        while not peek_is(endToken):
            arg = parseArg()
            if arg.get('named'):
                if usingNamed is False:
                    raise ValueError('Cannot mix positional and named arguments')
                usingNamed = True
                if arg['name'] in named:
                    raise ValueError(f"Duplicate argument: {arg['name']}")
                named[arg['name']] = arg['value']
            else:
                if usingNamed is True:
                    raise ValueError('Cannot mix positional and named arguments')
                usingNamed = False
                positional.append(arg['value'])
            if not match(','):
                break
        consume(endToken)
        return {'named': named} if usingNamed else {'positional': positional}
    def parseArg():
        t = peek()
        if t['type'] == 'identifier' and (peek(1) and peek(1)['type'] == ':'):
            name = consume('identifier')['value']
            consume(':')
            value = parseExpression()
            return {'named': True, 'name': name, 'value': value}
        return {'value': parseExpression()}
    def parseIdentifier():
        token = consume('identifier')
        return {'type': 'Identifier', 'name': token['value']}
    def parseNumberExpr():
        node = parseAdd()
        if match('?'):
            true_expr = parseNumberExpr()
            consume(':')
            false_expr = parseNumberExpr()
            node = {'type': 'TernaryExpr', 'test': node, 'consequent': true_expr, 'alternate': false_expr}
        return node
    def parseAdd():
        node = parseMul()
        while True:
            t = peek()
            if t and t['type'] in ('+', '-'):
                consume(t['type'])
                right = parseMul()
                node = {'type': 'BinaryExpr', 'operator': t['type'], 'left': node, 'right': right}
            else:
                break
        return node
    def parseMul():
        node = parseNumberPrimary()
        while True:
            t = peek()
            if t and t['type'] in ('*', '/'):
                consume(t['type'])
                right = parseNumberPrimary()
                node = {'type': 'BinaryExpr', 'operator': t['type'], 'left': node, 'right': right}
            else:
                break
        return node
    def parseNumberPrimary():
        t = peek()
        if t['type'] == 'number':
            consume('number')
            return {'type': 'NumberLiteral', 'value': t['value']}
        if t['type'] == '(':
            consume('(')
            expr = parseNumberExpr()
            consume(')')
            return expr
        if (t['type'] == 'identifier' and t['value'] == 'Math' and
            (peek(1) and peek(1)['type'] == '.') and
            (peek(2) and peek(2)['type'] == 'identifier') and
            peek(2)['value'] == 'PI'):
            consume('identifier')
            consume('.')
            consume('identifier')
            import math
            return {'type': 'NumberLiteral', 'value': math.pi}
        if t['type'] == 'identifier':
            if peek(1) and peek(1)['type'] == '(':
                return parseCallChain()
            id_ = parseIdentifier()
            if match('.'):
                member = parseIdentifier()
                node = {'type': 'MemberExpr', 'object': id_, 'property': member}
                while match('.'):
                    member = parseIdentifier()
                    node = {'type': 'MemberExpr', 'object': node, 'property': member}
                if match('('):
                    args = parseArgList(')')
                    node = {'type': 'CallExpr', 'callee': node, 'args': args}
                    while match('.'):
                        next_node = parseSingleCall()
                        next_node['input'] = node
                        node = next_node
                    return node
                return node
            return id_
        if t['type'] == 'boolean':
            consume('boolean')
            return {'type': 'NumberLiteral', 'value': 1 if t['value'] else 0}
        if t['type'] == 'null':
            consume('null')
            return {'type': 'NullLiteral', 'value': None}
        unexpected(t)
    return parseProgram()
