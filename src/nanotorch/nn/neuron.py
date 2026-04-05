import numpy as np

from nanotorch.value import Value
from nanotorch.nn.module import Module

class Neuron(Module):
    def __init__(self, input_dim, activation=None):
        # create random scalar weights
        self.weights: list[Value] = [Value(elem) for elem in np.random.randn(input_dim)]
        self.bias: Value = Value(np.random.randn())
        self.activation = activation

    # this is our forward method
    def __call__(self, x):
        # make sure x is an iterable
        try:
            iter(x)
        except TypeError:
            raise TypeError("Input must be an iterable of values")

        # neurons should operate on Values
        inputs = [elem if isinstance(elem, Value) else Value(elem) for elem in x]
        assert len(x) == len(self.weights), "Input and weights don't have the same shape!"

        # compute weighted sum with bias
        res = self.bias
        for x_i, w_i in zip(inputs, self.weights):
            res += w_i * x_i

        # do activation
        if self.activation:
            res = self.activation(res)
        return res

    # # get the trainable parameters
    # def parameters(self):
    #     return self.weights + [self.bias]

