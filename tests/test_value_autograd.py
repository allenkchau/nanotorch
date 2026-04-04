import pytest

from nanotorch.value import Value


@pytest.mark.parametrize(
    ("left", "right", "operator", "expected"),
    [
        (5.0, 6.0, "add", 11.0),
        (5.0, 6.0, "mul", 30.0),
    ],
)
def test_binary_ops_forward_build_expected_values(left, right, operator, expected):
    a = Value(left)
    b = Value(right)

    out = a + b if operator == "add" else a * b

    assert out.data == pytest.approx(expected)
    assert out.parents == (a, b)


def test_addition_local_backward_propagates_upstream_gradient():
    a = Value(5.0)
    b = Value(6.0)
    out = a + b

    out.grad = 60.0
    out._backward()

    assert a.grad == pytest.approx(60.0)
    assert b.grad == pytest.approx(60.0)


def test_multiplication_local_backward_uses_local_derivatives():
    a = Value(5.0)
    b = Value(6.0)
    out = a * b

    out.grad = 60.0
    out._backward()

    assert a.grad == pytest.approx(60.0 * b.data)
    assert b.grad == pytest.approx(60.0 * a.data)


def test_backward_accumulates_gradients_across_shared_subgraph():
    a = Value(5.0)
    b = Value(6.0)
    c = a * b
    out = c + a

    out.backward()

    assert a.grad == pytest.approx(b.data + 1.0)
    assert b.grad == pytest.approx(a.data)
    assert c.grad == pytest.approx(1.0)


def test_complex_graph_forward_and_backward_values_match_manual_derivatives():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b          # -6
    e = d + c          # 4
    f = a * e          # 8
    g = f / b          # -8/3
    h = g - a          # -14/3
    h.backward()

    assert d.data == pytest.approx(-6.0)
    assert e.data == pytest.approx(4.0)
    assert f.data == pytest.approx(8.0)
    assert g.data == pytest.approx(-8.0 / 3.0)
    assert h.data == pytest.approx(-14.0 / 3.0)

    # h = a*(a*b + c)/b - a = a^2 + a*c/b - a
    # dh/da = 2a + c/b - 1
    # dh/db = -a*c / b^2
    # dh/dc = a / b
    assert a.grad == pytest.approx(-1.0 / 3.0)
    assert b.grad == pytest.approx(-20.0 / 9.0)
    assert c.grad == pytest.approx(-2.0 / 3.0)


def test_radd_with_scalar_matches_forward_and_backward():
    a = Value(2.5)
    out = 3.0 + a

    out.backward()

    assert out.data == pytest.approx(5.5)
    assert a.grad == pytest.approx(1.0)


def test_rsub_with_scalar_matches_forward_and_backward():
    a = Value(2.0)
    out = 5.0 - a

    out.backward()

    assert out.data == pytest.approx(3.0)
    assert a.grad == pytest.approx(-1.0)


def test_rmul_with_scalar_matches_forward_and_backward():
    a = Value(-1.5)
    out = 4.0 * a

    out.backward()

    assert out.data == pytest.approx(-6.0)
    assert a.grad == pytest.approx(4.0)


def test_rtruediv_with_scalar_matches_forward_and_backward():
    a = Value(2.0)
    out = 8.0 / a

    out.backward()

    assert out.data == pytest.approx(4.0)
    assert a.grad == pytest.approx(-2.0)

