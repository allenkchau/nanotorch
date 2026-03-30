import pytest
from nanotorch.value import Value

def test_scalar_addition():
    a = Value(5)
    b = Value(6)
    c = a + b
    assert c.data == 11

