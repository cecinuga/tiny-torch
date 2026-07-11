from core.functions import sigmoid, tanh
from core.tensor import Tensor
from typing import override
from core.autograd.base import Function

class ReLUBackward(Function):
    """Gradient computation for ReLU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        grad_output = grad_output * (t > 0)
        return grad_output,

class SigmoidBackward(Function):
    """Gradient computation for Sigmoid activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]

        out = grad_output * (sigmoid(t.data)*sigmoid(1-t.data))
        return out,

class TanhBackward(Function):
    """Gradient computation for Tanh activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]

        out = grad_output * (1-tanh(t.data)**2)
        return out,

class GELUBackward(Function):
    """Gradient computation for GELU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        s = sigmoid(t * 1.702)
        local_grad = s * (1 + 1.702 * t * (1-s))
        return grad_output * local_grad,
