import numpy as np

class Value:
    def __init__(self, data):
        self.data = data
        # our gradient should initially be 0
        self.grad = 0
        # parents for 
        self.parents = ()
        self.op = ""
        # every node should have a _backward function; for leaf nodes we have a function that does nothing
        self._backward = lambda: None

    def __repr__(self):
        return f"Data: {self.data}, Grad: {self.grad}, Op: {self.op}"

    def __add__(self, other):
        res = Value(self.data + other.data)
        res.op = "+"
        res.parents = (self, other)

        # define the backward pass for addition
        def add_backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = add_backward
        return res

    def __sub__(self, other):
        res = Value(self.data - other.data)
        res.op = "-"
        res.parents = (self, other)

        # define the backward pass for subtraction
        def sub_backward():
            self.grad += res.grad
            other.grad += -res.grad

        res._backward = sub_backward
        return res

    def __mul__(self, other):
        res = Value(self.data * other.data)
        res.op = "*"
        res.parents = (self, other)

        # define the backward pass for multiplication
        def mul_backward():
            self.grad += res.grad * other.data
            other.grad += res.grad* self.data

        res._backward = mul_backward
        return res

    def __div__(self, other):
        res = Value(self.data / other.data)
        res.op = "/"
        res.parents = (self, other)

        # define the backward pass for division
        def div_backward():
            self.grad += res.grad
            other.grad += -res.grad

        res._backward = div_backward
        return res

    # backprop algo
    def backward(self):
        # if this is our output or final loss node, dL/dL = 1
        self.grad = 1
        
        # first we build a topological ordering from output node
        def dfs(node):
            order = []
            if not node:
                return
            for parent in node.parents:
                dfs(parent)
            return order

        # we loop over nodes and call _backward for each node in the graph
        for node in nodes:
            node._backward()
        

    
    
