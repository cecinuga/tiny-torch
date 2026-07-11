import numpy as np
from typing import override
from core.tensor import Tensor
from core.utils import unbroadcast
from core.autograd.base import Function

class AddBackward(Function):
    """Gradient computation for addition."""

    @override
    def apply(self, grad_output: Tensor)-> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = grad_output

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class SubBackward(Function):
    """Gradient computation for subtraction."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = grad_output

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(-grad_output.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class MulBackward(Function):
    """Gradient computation for multiplication."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = grad_output

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * b.data, a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * a.data, b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class DivBackward(Function):
    """Gradient computation for division."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = grad_output

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * (1/b.data), a.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * -a.data/(b.data**2), b.shape)

        return Tensor(grad_a), Tensor(grad_b)

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = grad_output

        # Aligns shapes by transposing the partner matrix
        if a.requires_grad:
            b_T = np.swapaxes(b.data, -2, -1)
            grad_a = np.matmul(grad_output.data, b_T)

        if b.requires_grad:
            a_T = np.swapaxes(a.data, -2, -1)
            grad_b = np.matmul(a_T, grad_output.data)

        return Tensor(grad_a), Tensor(grad_b)

class SumBackward(Function):
    """Gradient computation for sum reduction."""

    def __init__(self, x: Tensor, axis:int|None = -1, keepdims:bool = True):
        super().__init__(x)
        self.axis:int|None = axis
        self.keepdims:bool = keepdims

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        grad = grad_output.data

        if not t.requires_grad:
            return grad_output,

        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, self.axis)

        return Tensor(np.broadcast_to(grad, t.shape)),

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        if not a.requires_grad:
            return grad_output,

        return Tensor(np.reshape(grad_output.data, a.shape)),

class TransposeBackward(Function):
    """Gradient computation for transpose."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        return Tensor(np.transpose(grad_output.data)),
