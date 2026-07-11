import numpy as np
from typing import override
from core.tensor import Tensor
from core.autograd.base import Function

class AddBackward(Function):
    """Gradient computation for addition."""

    @override
    def apply(self, grad_output: Tensor)-> tuple[Tensor, Tensor]:
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, grad_output

class SubBackward(Function):
    """Gradient computation for subtraction."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, -grad_output

class MulBackward(Function):
    """Gradient computation for multiplication."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = grad_output.data * b.data

        if b.requires_grad:
            grad_b = grad_output.data * a.data

        return Tensor(grad_a), Tensor(grad_b)

class DivBackward(Function):
    """Gradient computation for division."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = grad_output.data * (1/b.data)

        if b.requires_grad:
            grad_b = grad_output.data * -a.data/(b.data**2)

        return Tensor(grad_a), Tensor(grad_b)

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a, b = self.saved_tensors
        grad_a = grad_b = []

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

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        t, = self.saved_tensors
        res = Tensor(np.broadcast_to(grad_output.data, t.shape))
        return res,

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        a, = self.saved_tensors
        if a.requires_grad:
            return grad_output.reshape(a.shape),
        return Tensor([]),

class TransposeBackward(Function):
    """Gradient computation for transpose."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        return grad_output.transpose(),
