"""
uv run pytest -v tests/
"""


from nanotorch.value import Value

def test_scalar_addition_forward():
    a = Value(5)
    b = Value(6)
    c = a + b
    assert c.data == 11
    assert a in c.parents and b in c.parents
    print(c)

def test_scalar_multiplication_forward():
    a = Value(5)
    b = Value(6)
    c = a * b
    assert c.data == 30
    assert a in c.parents and b in c.parents
    print(c)

def test_scalar_addition_local_backward():
    a = Value(5)
    b = Value(6)
    c = a + b

    # arbitrarily set the grad to a number and see if the parent grads follow the math
    c.grad = 60
    c._backward()
    assert a.grad == 60
    assert b.grad == 60
    assert a in c.parents and b in c.parents

def test_scalar_multiplication_local_backward():
    a = Value(5)
    b = Value(6)
    c = a * b

    # arbitrarily set the grad to a number and see if the parent grads follow the math
    c.grad = 60
    c._backward()
    assert a.grad == 360
    assert b.grad == 300
    assert a in c.parents and b in c.parents

def test_scalar_backward():
    a = Value(5)
    b = Value(6)
    c = a * b
    d = c + a

    d.backward()
    assert a.grad == b.data + 1
    assert b.grad == a.data
    assert c.grad == 1

def test_complex_scalar_backprop():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b          # -6
    e = d + c          # 4
    f = a * e          # 8
    g = f / b          # -8/3
    h = g - a          # -14/3

    h.backward()

    # forward checks
    assert abs(d.data - (-6.0)) < 1e-9
    assert abs(e.data - 4.0) < 1e-9
    assert abs(f.data - 8.0) < 1e-9
    assert abs(g.data - (-8.0 / 3.0)) < 1e-9
    assert abs(h.data - (-14.0 / 3.0)) < 1e-9

    # gradient checks
    # h = a*(a*b + c)/b - a
    #   = a^2 + a*c/b - a
    #
    # dh/da = 2a + c/b - 1 = 4 - 10/3 - 1 = -1/3
    # dh/db = -a*c / b^2 = -(2*10)/9 = -20/9
    # dh/dc = a/b = -2/3

    assert abs(a.grad - (-1.0 / 3.0)) < 1e-9
    assert abs(b.grad - (-20.0 / 9.0)) < 1e-9
    assert abs(c.grad - (-2.0 / 3.0)) < 1e-9

