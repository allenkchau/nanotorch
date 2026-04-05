import numpy as np
import pytest

from nanotorch.nn.layer import Layer
from nanotorch.nn.mlp import MLP
from nanotorch.nn.neuron import Neuron
from nanotorch.value import Value


@pytest.mark.parametrize("input_dim", [1, 3, 7])
def test_neuron_parameters_count_via_module_base_class(input_dim):
    np.random.seed(0)
    neuron = Neuron(input_dim=input_dim)

    params = neuron.parameters()

    assert len(params) == input_dim + 1
    assert all(isinstance(param, Value) for param in params)


@pytest.mark.parametrize(
    ("input_dim", "output_dim"),
    [
        (1, 1),
        (3, 2),
        (5, 4),
    ],
)
def test_layer_parameters_count_via_module_base_class(input_dim, output_dim):
    np.random.seed(1)
    layer = Layer(input_dim=input_dim, output_dim=output_dim)

    params = layer.parameters()
    expected_count = output_dim * (input_dim + 1)

    assert len(params) == expected_count
    assert all(isinstance(param, Value) for param in params)


@pytest.mark.parametrize(
    "layer_sizes",
    [
        [3, 1],
        [3, 4, 2],
        [5, 8, 6, 4, 1],
    ],
)
def test_mlp_parameters_count_via_recursive_module_traversal(layer_sizes):
    np.random.seed(2)
    mlp = MLP(layer_sizes=layer_sizes)

    params = mlp.parameters()
    expected_count = sum(
        out_dim * (in_dim + 1)
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
    )

    assert len(params) == expected_count
    assert all(isinstance(param, Value) for param in params)


def test_zero_grad_resets_all_neuron_parameter_gradients():
    np.random.seed(3)
    neuron = Neuron(input_dim=3)
    x = [1.0, -2.0, 0.5]

    out = neuron(x)
    out.backward()
    assert any(param.grad != 0 for param in neuron.parameters())

    neuron.zero_grad()

    assert all(param.grad == 0 for param in neuron.parameters())


def test_zero_grad_resets_gradients_recursively_for_mlp():
    np.random.seed(4)
    mlp = MLP(layer_sizes=[3, 4, 2])
    x = [0.5, -1.0, 2.0]

    outputs = mlp(x)
    loss = outputs[0] + outputs[1]
    loss.backward()
    assert any(param.grad != 0 for param in mlp.parameters())

    mlp.zero_grad()

    assert all(param.grad == 0 for param in mlp.parameters())
