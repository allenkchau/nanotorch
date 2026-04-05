from nanotorch.value import Value


class SGD:
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters: list[Value] = list(parameters)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
