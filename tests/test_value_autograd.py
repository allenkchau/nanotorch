from calendar import c
import pytest
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

def test_global_backward():
    a = Value(5)
    b = Value(6)
    c = Value(7)
    d = a * b + c
    assert a.grad == 
    assert b.grad == 
    assert c.grad == 

