from nanotorch.nn.neuron import Neuron
from nanotorch.nn.module import Module

class Layer(Module):
    def __init__(self, input_dim, output_dim, activation=None):
        self.neurons: list[Neuron] = [Neuron(input_dim=input_dim, activation=activation) for _ in range(output_dim)]

    def __call__(self, x):
        outputs = []
        for neuron in self.neurons:
            output = neuron(x)
            outputs.append(output)
        return outputs

    # # get the trainable parameters
    # def parameters(self):
    #     return [param for neuron in self.neurons for param in neuron.parameters()]


