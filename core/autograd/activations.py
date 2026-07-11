from core.functions import sigmoid, tanh, softmax
from core.tensor import Tensor
from typing import override
from core.autograd.base import Function

class ReLUBackward(Function):
    """Gradient computation for ReLU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        local_grad = (t > 0)

        return grad_output * local_grad,

class SigmoidBackward(Function):
    """Gradient computation for Sigmoid activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        s = sigmoid(t.data)
        local_grad = s*(1-s)

        return grad_output * local_grad,

class TanhBackward(Function):
    """Gradient computation for Tanh activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        local_grad = (1-tanh(t.data)**2)

        return grad_output * local_grad,

class GELUBackward(Function):
    """Gradient computation for GELU activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = self.saved_tensors[0]
        s = sigmoid(t.data * 1.702)
        local_grad: Tensor = s * (1 + 1.702 * t * (1-s))

        return grad_output * local_grad,

class SoftmaxBackward(Function):
    """Gradient computation for Softmax activation."""

    @override
    def apply(self, grad_output:Tensor) -> tuple[Tensor, ...]:
        t = softmax(self.saved_tensors[0].data)
        dot = (grad_output*t).sum(axis=-1, keepdims=True)
        return t * (grad_output - dot),
