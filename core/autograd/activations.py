import numpy as np
from core.functions import sigmoid, tanh, softmax
from core.tensor import Tensor
from typing import override
from core.autograd.base import Function

class ReLUBackward(Function):
    """Gradient computation for ReLU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        local_grad = (t > 0)

        return Tensor(grad_output.data * local_grad.data),

class SigmoidBackward(Function):
    """Gradient computation for Sigmoid activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = sigmoid(t.data)
        local_grad = s*(1-s)

        return Tensor(grad_output.data * local_grad),

class TanhBackward(Function):
    """Gradient computation for Tanh activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        local_grad = (1-tanh(t.data)**2)

        return Tensor(grad_output.data * local_grad),

class GELUBackward(Function):
    """Gradient computation for GELU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = sigmoid(t.data * 1.702)
        local_grad = s * (1 + 1.702 * t.data * (1-s))

        return Tensor(grad_output.data * local_grad),

class SoftmaxBackward(Function):
    """Gradient computation for Softmax activation."""

    def __init__(self, x: Tensor, dim:int):
        super().__init__(x)
        self.dim:int = dim

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        s = softmax(t.data)
        dot = np.sum(grad_output.data*s, axis=self.dim, keepdims=True)
        return Tensor(s * (grad_output.data - dot)),
