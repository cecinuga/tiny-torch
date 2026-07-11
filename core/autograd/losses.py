import numpy as np
from core.tensor import Tensor
from core.functions import softmax
from core.autograd import Function
from typing import override

class MSELossBackward(Function):
    """Gradient computation for MSE loss."""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        predictions, targets = self.saved_tensors
        err = (predictions.data - targets.data)
        local_grad = (2*err)/len(predictions.data)

        return Tensor(grad_output.data * local_grad),

class CrossEntropyBackward(Function):
    """Gradient computation for Cross Entropy loss"""

    @override
    def apply(self, grad_output: Tensor) -> tuple[Tensor, ...]:
        logits, targets = self.saved_tensors
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # dL/dlogits = (softmax(logits) - onehot(targets)) / B
        probs = softmax(logits.data, dim=-1)
        probs[np.arange(batch_size), target_indices] -= 1.0
        local_grad: np.ndarray = probs / batch_size

        return Tensor(grad_output.data * local_grad),
