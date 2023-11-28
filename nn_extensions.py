import torch.nn as nn

from functools import partial
from inspect import signature, _ParameterKind
from torch import Tensor
from typing import Callable



class Apply(nn.Module):
    """
    Wrapper for a function as a torch module.
    """

    def __init__(self, fn: Callable, **fn_kwargs):
        super().__init__()
        self.forward = partial(fn, **fn_kwargs)


class BatchNormNd(nn.Module):
    """
    N-dimensional batch normalization.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.bn = None
        self.kwargs = kwargs

    def forward(self, input: Tensor) -> Tensor:
        if self.bn is None:
            if input.ndim == 4:
                self.bn = nn.BatchNorm2d(input.shape[1], **self.kwargs)
            elif input.ndim == 5:
                self.bn = nn.BatchNorm3d(input.shape[1], **self.kwargs)
            else:
                self.bn = nn.BatchNorm1d(input.shape[1], **self.kwargs)

        return self.bn(input)


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


class Dummy(nn.Module):
    """
    Dummy module that returns a tensor of shape (N, 0).
    """

    def forward(self, input: Tensor):
        return input.flatten(start_dim=1)[:, []]


class VariableKwargs(nn.Module):
    """
    Wrapper to allow module to take in variable keyword arguments.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def __getattr__(self, name: str):
        if name == 'module':
            return super().__getattr__(name)
        else:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        parameters = signature(self.module.forward).parameters.values()
        if any(p.kind == _ParameterKind.VAR_KEYWORD for p in parameters):
            module_kwargs = kwargs
        else:
            module_kwargs = {
                key: value for key, value in kwargs.items()
                if key in signature(self.module.forward).parameters.keys()
            }

        return self.module(*args, **module_kwargs)



# Monkey-patch nn.Module to support '+' operator for chaining modules
nn.Module.__add__ = Chain.__add__
nn.Module.__radd__ = Chain.__radd__
