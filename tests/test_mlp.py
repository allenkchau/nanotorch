import numpy as np
import pytest

from nanotorch.nn.layer import Layer
from nanotorch.nn.mlp import MLP
from nanotorch.value import Value


def test_mlp_initialization_builds_expected_number_of_layers():
    mlp = MLP(layer_sizes=[3, 4, 2])

    assert len(mlp.layers) == 2
    assert all(isinstance(layer, Layer) for layer in mlp.layers)


def test_mlp_applies_activation_to_hidden_layers_but_not_output_layer():
    def identity(v: Value) -> Value:
        return v

    mlp = MLP(layer_sizes=[2, 3, 1], activation=identity)

    assert all(neuron.activation is identity for neuron in mlp.layers[0].neurons)
    assert all(neuron.activation is None for neuron in mlp.layers[-1].neurons)


def test_mlp_forward_returns_output_list_of_values():
    np.random.seed(0)
    mlp = MLP(layer_sizes=[3, 4, 2])
    x = [1.0, -2.0, 3.0]

    out = mlp(x)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(elem, Value) for elem in out)


def test_mlp_forward_matches_manual_two_layer_computation_without_activation():
    np.random.seed(7)
    mlp = MLP(layer_sizes=[2, 3, 1], activation=None)
    x = [1.5, -0.5]

    hidden = []
    for neuron in mlp.layers[0].neurons:
        h = neuron.bias.data + sum(w.data * x_i for w, x_i in zip(neuron.weights, x))
        hidden.append(h)

    expected_output = mlp.layers[1].neurons[0].bias.data + sum(
        w.data * h_i for w, h_i in zip(mlp.layers[1].neurons[0].weights, hidden)
    )

    out = mlp(x)

    assert len(out) == 1
    assert out[0].data == pytest.approx(expected_output)


def test_mlp_forward_uses_hidden_activation_but_not_output_activation():
    def scale_by_two(v: Value) -> Value:
        return Value(v.data * 2)

    np.random.seed(13)
    mlp = MLP(layer_sizes=[2, 2, 1], activation=scale_by_two)
    x = [0.5, -1.5]

    hidden_after_activation = []
    for neuron in mlp.layers[0].neurons:
        pre = neuron.bias.data + sum(w.data * x_i for w, x_i in zip(neuron.weights, x))
        hidden_after_activation.append(pre * 2)

    expected_output = mlp.layers[1].neurons[0].bias.data + sum(
        w.data * h_i
        for w, h_i in zip(mlp.layers[1].neurons[0].weights, hidden_after_activation)
    )

    out = mlp(x)
    assert out[0].data == pytest.approx(expected_output)


def test_mlp_parameters_flattens_all_layer_parameters_in_order():
    np.random.seed(11)
    layer_sizes = [3, 4, 2]
    mlp = MLP(layer_sizes=layer_sizes)

    params = mlp.parameters()
    expected_count = sum(
        out_dim * (in_dim + 1)
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
    )

    assert len(params) == expected_count
    expected = [param for layer in mlp.layers for param in layer.parameters()]
    assert params == expected


def test_mlp_forward_propagates_non_iterable_input_error():
    mlp = MLP(layer_sizes=[2, 3, 1])

    with pytest.raises(TypeError, match="Input must be an iterable of values"):
        mlp(123)


def test_mlp_forward_propagates_shape_mismatch_error():
    mlp = MLP(layer_sizes=[3, 2, 1])

    with pytest.raises(AssertionError, match="same shape"):
        mlp([1.0, 2.0])


def test_mlp_backward_populates_gradients_for_all_parameters():
    np.random.seed(19)
    mlp = MLP(layer_sizes=[2, 3, 2])
    x = [1.25, -0.75]

    outputs = mlp(x)
    loss = outputs[0]
    for out in outputs[1:]:
        loss = loss + out
    loss.backward()

    for param in mlp.parameters():
        assert isinstance(param.grad, (int, float, np.floating))


def test_large_mlp_forward_and_backward_matches_numerical_gradients():
    np.random.seed(123)
    layer_sizes = [5, 8, 6, 4, 1]
    mlp = MLP(layer_sizes=layer_sizes)
    x = [0.5, -1.0, 2.0, -0.25, 1.5]

    def loss_data() -> float:
        outputs = mlp(x)
        return float(outputs[0].data)

    outputs = mlp(x)
    assert len(outputs) == 1
    assert all(isinstance(out, Value) for out in outputs)

    params = mlp.parameters()
    expected_param_count = sum(
        out_dim * (in_dim + 1)
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
    )
    assert len(params) == expected_param_count

    loss = outputs[0]
    loss.backward()

    analytic_grads = [float(param.grad) for param in params]

    epsilon = 1e-6
    indices_to_check = [0, len(params) // 2, len(params) - 1]
    for idx in indices_to_check:
        param = params[idx]
        original = float(param.data)

        param.data = original + epsilon
        loss_plus = loss_data()

        param.data = original - epsilon
        loss_minus = loss_data()

        param.data = original
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

        assert analytic_grads[idx] == pytest.approx(numerical_grad, rel=1e-4, abs=1e-5)
