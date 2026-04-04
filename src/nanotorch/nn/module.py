"""
Module abstraction for neural network components.

A Module is the base class for all trainable components in nanotorch.
It provides a unified interface for organizing parameters and composing
models hierarchically.

The parameters() method recursively traverses this structure to return
all trainable parameters in the model.
"""

class Module:
    def __init__(self):
        pass

    def parameters(self):
        pass
