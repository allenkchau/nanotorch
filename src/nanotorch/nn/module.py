"""
Module abstraction for neural network components.

A Module is the base class for all trainable components in nanotorch.
It provides a unified interface for organizing parameters and composing
models hierarchically.

The parameters() method recursively traverses this structure to return
all trainable parameters in the model.
"""

from nanotorch.value import Value

class Module:
    # we might not need init for Module
    # def __init__(self):
        

    # recursively collect all parameters in the module
    def parameters(self):

        def params_helper(obj):
            res = []
            if isinstance(obj, Value):
                res.append(obj)
            elif isinstance(obj, Module):
                for k, v in obj.__dict__.items():
                    res.extend(params_helper(v))
            elif isinstance(obj, list):
                for elem in obj:
                    res.extend(params_helper(elem))
            return res
    
        # call the helper function
        params = params_helper(self)
        return params

    # gives us a cleaner piece in the training loop to set all gradients to zero
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    

