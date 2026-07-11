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
        grad_a = grad_b = np.array([])

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
        grad_a = grad_b = np.array([])

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
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * b.data, b.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * a.data, a.shape)

        return Tensor(grad_a), Tensor(grad_b)

class DivBackward(Function):
    """Gradient computation for division."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        inv = (1/b.data)
        if a.requires_grad:
            grad_a = unbroadcast(grad_output.data * inv, b.shape)
        if b.requires_grad:
            grad_b = unbroadcast(grad_output.data * -a.data*(inv**2), a.shape)

        return Tensor(grad_a), Tensor(grad_b)

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

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

        if not t.requires_grad:
            return Tensor(np.array([])),

        grad = grad_output.data
        if not self.keepdims and self.axis is not None:
            grad = np.expand_dims(grad, self.axis)

        return Tensor(np.broadcast_to(grad, t.shape)),

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        out = np.array([])

        if a.requires_grad:
            out = np.reshape(grad_output.data, a.shape),

        return Tensor(out),

class TransposeBackward(Function):
    """Gradient computation for transpose."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        out = np.array([])

        if t.requires_grad:
            out = np.transpose(grad_output.data),

        return Tensor(out),
