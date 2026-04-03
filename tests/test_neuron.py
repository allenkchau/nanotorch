import numpy as np
import pytest

from nanotorch.nn.neuron import Neuron
from nanotorch.value import Value


def test_neuron_initialization_creates_value_parameters():
    np.random.seed(0)
    neuron = Neuron(input_dim=4)

    assert len(neuron.weights) == 4
    assert all(isinstance(w, Value) for w in neuron.weights)
    assert isinstance(neuron.bias, Value)
    assert neuron.activation is None


def test_neuron_parameters_returns_weights_plus_bias_in_order():
    np.random.seed(1)
    neuron = Neuron(input_dim=3)

    params = neuron.parameters()
    assert len(params) == 4
    assert params[:-1] == neuron.weights
    assert params[-1] is neuron.bias


def test_neuron_forward_matches_manual_weighted_sum_for_float_inputs():
    np.random.seed(42)
    neuron = Neuron(input_dim=3)
    x = [1.0, -2.0, 0.5]

    out = neuron(x)
    expected = neuron.bias.data + sum(
        w.data * x_i for w, x_i in zip(neuron.weights, x)
    )

    assert isinstance(out, Value)
    assert out.data == pytest.approx(expected)


def test_neuron_forward_accepts_value_inputs_and_matches_float_inputs():
    np.random.seed(7)
    neuron = Neuron(input_dim=2)
    raw = [3.0, -4.0]

    out_from_floats = neuron(raw)
    out_from_values = neuron([Value(v) for v in raw])

    assert out_from_values.data == pytest.approx(out_from_floats.data)


def test_neuron_non_iterable_input_raises_type_error():
    neuron = Neuron(input_dim=2)

    with pytest.raises(TypeError, match="Input must be an iterable of values"):
        neuron(123)


def test_neuron_shape_mismatch_raises_assertion_error():
    neuron = Neuron(input_dim=3)

    with pytest.raises(AssertionError, match="same shape"):
        neuron([1.0, 2.0])


def test_neuron_activation_is_applied():
    def scale_by_two(v: Value) -> Value:
        return Value(v.data * 2)

    np.random.seed(9)
    neuron = Neuron(input_dim=2, activation=scale_by_two)
    x = [0.25, -0.75]

    weighted_sum = neuron.bias.data + sum(
        w.data * x_i for w, x_i in zip(neuron.weights, x)
    )
    out = neuron(x)

    assert out.data == pytest.approx(weighted_sum * 2)


def test_neuron_backward_populates_parameter_grads_correctly():
    np.random.seed(11)
    neuron = Neuron(input_dim=3)
    x = [2.0, -3.0, 4.0]

    out = neuron(x)
    out.backward()

    for w, x_i in zip(neuron.weights, x):
        assert w.grad == pytest.approx(x_i)
    assert neuron.bias.grad == pytest.approx(1.0)
