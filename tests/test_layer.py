import numpy as np
import pytest

from nanotorch.nn.layer import Layer
from nanotorch.nn.neuron import Neuron
from nanotorch.value import Value


def test_layer_initialization_creates_expected_number_of_neurons():
    np.random.seed(0)
    layer = Layer(input_dim=3, output_dim=4)

    assert len(layer.neurons) == 4
    assert all(isinstance(neuron, Neuron) for neuron in layer.neurons)


def test_layer_passes_activation_to_each_neuron():
    def identity(v: Value) -> Value:
        return v

    layer = Layer(input_dim=2, output_dim=3, activation=identity)

    assert all(neuron.activation is identity for neuron in layer.neurons)


def test_layer_forward_returns_one_value_per_neuron():
    np.random.seed(42)
    layer = Layer(input_dim=3, output_dim=2)
    x = [1.5, -2.0, 0.25]

    outputs = layer(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert all(isinstance(out, Value) for out in outputs)


def test_layer_forward_matches_manual_neuron_computation():
    np.random.seed(7)
    layer = Layer(input_dim=2, output_dim=3)
    x = [2.0, -4.0]

    outputs = layer(x)
    expected = [
        neuron.bias.data + sum(w.data * x_i for w, x_i in zip(neuron.weights, x))
        for neuron in layer.neurons
    ]

    assert [out.data for out in outputs] == pytest.approx(expected)


def test_layer_parameters_flattens_all_neuron_parameters_in_order():
    np.random.seed(11)
    input_dim = 3
    output_dim = 2
    layer = Layer(input_dim=input_dim, output_dim=output_dim)

    params = layer.parameters()

    assert len(params) == output_dim * (input_dim + 1)
    expected = [param for neuron in layer.neurons for param in neuron.parameters()]
    assert params == expected


def test_layer_forward_propagates_non_iterable_input_error():
    layer = Layer(input_dim=2, output_dim=2)

    with pytest.raises(TypeError, match="Input must be an iterable of values"):
        layer(123)


def test_layer_forward_propagates_shape_mismatch_error():
    layer = Layer(input_dim=3, output_dim=2)

    with pytest.raises(AssertionError, match="same shape"):
        layer([1.0, 2.0])


def test_layer_backward_sets_gradients_for_each_neuron_parameters():
    np.random.seed(19)
    layer = Layer(input_dim=3, output_dim=2)
    x = [1.0, -2.0, 3.0]

    outputs = layer(x)
    loss = outputs[0]
    for out in outputs[1:]:
        loss = loss + out
    loss.backward()

    for neuron in layer.neurons:
        for w, x_i in zip(neuron.weights, x):
            assert w.grad == pytest.approx(x_i)
        assert neuron.bias.grad == pytest.approx(1.0)
