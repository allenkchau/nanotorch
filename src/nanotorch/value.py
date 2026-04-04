import numpy as np

class Value:
    def __init__(self, data):
        self.data = data
        # our gradient should initially be 0
        self.grad = 0
        # parents for leaves are empty
        self.parents = ()
        self.op = ""
        # every node should have a _backward function; for leaf nodes we have a function that does nothing
        self._backward = lambda: None

    def __repr__(self):
        return f"Data: {self.data}, Grad: {self.grad}, Op: {self.op}"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value(self.data + other.data)
        res.op = "+"
        res.parents = (self, other)

        # define the backward pass for addition
        def add_backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = add_backward
        return res

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value(self.data - other.data)
        res.op = "-"
        res.parents = (self, other)

        # define the backward pass for subtraction
        def sub_backward():
            self.grad += res.grad
            other.grad += -res.grad

        res._backward = sub_backward
        return res

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value(self.data * other.data)
        res.op = "*"
        res.parents = (self, other)

        # define the backward pass for multiplication
        def mul_backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data

        res._backward = mul_backward
        return res

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value(self.data / other.data)
        res.op = "/"
        res.parents = (self, other)

        # define the backward pass for division
        def div_backward():
            self.grad += res.grad * (1 / other.data)
            other.grad += res.grad * (-self.data / other.data**2)

        res._backward = div_backward
        return res

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return other / self

    
    # activations are also just operations in our computational graph
    def relu(self):
        res = Value(max(0, self.data))
        res.op = "relu"
        res.parents = (self,)
        def relu_backward():
            if self.data > 0:
                self.grad += res.grad

        res._backward = relu_backward
        return res


    # backprop algo
    def backward(self):
        # first we build the graph order
        # we don't want to visit the same node again when traversing the graph
        order = []
        visited = set()

        # dfs helper
        def dfs(node):
            visited.add(node)
            for parent in node.parents:
                if parent not in visited:
                    dfs(parent)
            order.append(node)
            return order

        # build graph
        dfs(self)

        # if this is our output or final loss node, dL/dL = 1
        self.grad = 1

        # run backward
        # we process over nodes in reverse topological order and call _backward for each node in the graph
        for node in order[::-1]:
            node._backward()
        

    
    
