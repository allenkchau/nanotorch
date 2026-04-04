from nanotorch.value import Value

def relu(x: Value) -> Value:
    return x.relu()

def tanh(x: Value) -> Value:
    return x.tanh()

def gelu(x: Value) -> Value:
    return x.gelu()

def silu(x: Value) -> Value:
    return x.silu()
