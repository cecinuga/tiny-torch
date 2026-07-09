import numpy as np
from typing import override
from core.autograd.base import Function

class ReLUBackward(Function):
    """Gradient computation for ReLU activation."""

    @override
    def apply(self, grad_output:np.ndarray) -> tuple[np.ndarray, ...]:
        for t in self.saved_tensors:
            grad_output = grad_output * (t > 0)
        return grad_output,
