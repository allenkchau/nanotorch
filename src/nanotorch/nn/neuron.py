import numpy as np

from nanotorch.value import Value

class Neuron:
    def __init__(self, input_dim, activation=None):
        # create random scalar weights
        self.weights: list[Value] = [Value(elem) for elem in np.random.randn(input_dim)]
        self.bias: Value = Value(np.random.randn())
        self.activation = activation

    # this is our forward method
    def __call__(self, x: list[Value]):
        assert len(x) == len(self.weights), "Input and weights don't have the same shape!"

        # compute weighted sum with bias
        res = self.bias
        for i, w in zip(x, self.weights):
            res += w * i

        # do activation
        if self.activation:
            res = self.activation(res)
        return res

    # get the trainable parameters
    def parameters(self):
        return self.weights + [self.bias]

