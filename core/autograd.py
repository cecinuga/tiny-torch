import numpy as np
from core.tensor import Tensor

def enable_autograd(quiet=False) -> None:
    pass

class Function:
    def __init__(self, *tensors: Tensor):
        # The Memory Cost
        self.saved_tensors:tuple[Tensor, ...] = tensors
        # The Graph Structure
        self.next_functions = []

    def apply(self, grad_output: Tensor):
        """Compute gradients for the inputs."""
        raise NotImplementedError()

class AddBackward(Function):
    """Gradient computation for addition."""

    def apply(self, grad_output:Tensor):
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, grad_output

class SubBackward(Function):
    """Gradient computation for subtraction."""

    def apply(self, grad_output:Tensor):
        # Gradient flows equally to both inputs
        # 1 * grad_output
        return grad_output, grad_output

class MulBackward(Function):
    """Gradient computation for multiplication."""

    def apply(self, grad_output:Tensor):
        a, b = self.saved_tensors
        # Retrieve from memory
        # Scale by the 'other' input
        grad_a = grad_output * b.data
        grad_b = grad_output * a.data

        return grad_a, grad_b

class DivBackward(Function):
    """Gradient computation for division."""

    def apply(self, grad_output:Tensor):
        a, b = self.saved_tensors
        grad_a = grad_output * (1/b.data)
        grad_b = grad_output * -a/(b**2)

class MatmulBackward(Function):
    """Gradient computation for matrix multiplication."""

    def apply(self, grad_output:Tensor):
        a, b = self.saved_tensors

        # Aligns shapes by transposing the partner matrix
        grad_a = np.matmul(grad_output.data, b.data.T)
        grad_b = np.matmul(a.data.T, grad_output.data)

        return grad_a, grad_b

class SumBackward(Function):
    """Gradient computation for sum reduction."""

    def apply(self, grad_output:Tensor):
        return grad_output

class ReshapeBackward(Function):
    """Gradient computation for reshape operation."""

    def apply(self, grad_output:Tensor):
        a, = self.saved_tensors
        return grad_output.reshape(a.shape)

class TransposeBackward(Function):
    """Gradient computation for transpose."""

    def apply(self, grad_output:Tensor):
        return grad_output.transpose()
