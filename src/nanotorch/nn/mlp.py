from nanotorch.nn.layer import Layer

class MLP:
    def __init__(self, layer_sizes, activation=None):
        self.layers: list[Layer] = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # there are len(layer_sizes) - 1 actual layers so -2 to get the last layer idx
            if i == len(layer_sizes) - 2:
                layer = Layer(input_dim=in_dim, output_dim=out_dim, activation=None)
            else:
                layer = Layer(input_dim=in_dim, output_dim=out_dim, activation=activation)
            self.layers.append(layer)


    def __call__(self, x):
        curr = x
        for layer in self.layers:
            output = layer(curr)
            curr = output
        return curr

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
