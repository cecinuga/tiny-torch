from core.tensor import Tensor
from core.activations import Sigmoid
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
        s = Sigmoid()
        t = self.saved_tensors[0]

        out = grad_output * (s(t)*s(1-t))
        return out,
