from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from core.tensor import Tensor

def enable_autograd(quiet:bool=False) -> None:
    pass

class Function:
    def __init__(self, *tensors: Tensor):
        # The Memory Cost
        self.saved_tensors:tuple[Tensor, ...] = tensors
        # The Graph Structure
        self.next_functions:list[Function|None] = [t._grad_fn for t in tensors]

    def apply(self, grad_output: np.ndarray)-> tuple[np.ndarray, ...]:
        """Compute gradients for the inputs."""
        raise NotImplementedError()

class AddBackward(Function):
    """Gradient computation for addition."""

    @override
    def apply(self, grad_output:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, grad_output

class SubBackward(Function):
    """Gradient computation for subtraction."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, grad_output

class MulBackward(Function):
    """Gradient computation for multiplication."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = grad_output * b.data

        if b.requires_grad:
            grad_b = grad_output * a.data

        return grad_a, grad_b

class DivBackward(Function):
    """Gradient computation for division."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        if a.requires_grad:
            grad_a = grad_output * (1/b.data)

        if b.requires_grad:
            grad_b = grad_output * -a/(b**2)

        return grad_a, grad_b

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.saved_tensors
        grad_a = grad_b = np.array([])

        # Aligns shapes by transposing the partner matrix
        if a.requires_grad:
            b_T = np.swapaxes(b.data, -2, -1)
            grad_a = np.matmul(grad_output, b_T)

        if b.requires_grad:
            a_T = np.swapaxes(a.data, -2, -1)
            grad_b = np.matmul(a_T, grad_output)

        return grad_a, grad_b

class SumBackward(Function):
    """Gradient computation for sum reduction."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, ...]:
        return grad_output,

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, ...]:
        a, = self.saved_tensors
        if a.requires_grad:
            return grad_output.reshape(a.shape),
        return np.array([]),

class TransposeBackward(Function):
    """Gradient computation for transpose."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, ...]:
        return grad_output.transpose(),
