import numpy as np

class Value:
    def __init__(self, data):
        self.data = data
        # our gradient should initially be 0
        self.grad = 0
        # parents for 
        self.parents = ()
        self.op = ""

    def __add__(self, other):
        c = Value(self.data + other.data)
        c.op = "+"
        c.parents = ()
        return c

    def __mul__(self, a, b):
        c = Value(a.data * b.data)
        return c

    
    
