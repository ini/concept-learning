import torch.nn as nn
from typing import Callable



class Apply(nn.Module):
    """
    Wrapper for a function as a torch module.
    """

    def __init__(self, fn: Callable, **fn_kwargs):
        super().__init__()
        self.fn = fn
        self.fn_kwargs = fn_kwargs

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs, **self.fn_kwargs)


class Chain(nn.Sequential):
    """
    Sequential module that supports additional arguments in forward pass.
    """

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input

    def __add__(module_1: nn.Module, module_2: nn.Module):
        if not isinstance(module_1, Chain):
            module_1 = Chain(module_1)
        if not isinstance(module_2, Chain):
            module_2 = Chain(module_2)

        return Chain(*module_1, *module_2)

    __radd__ = __add__



# Monkey-patch nn.Module to support '+' operator for chaining modules
nn.Module.__add__ = Chain.__add__
