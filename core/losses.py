import numpy as np
from core.tensor import Tensor

def log_softmax(x: Tensor, dim:int=-1) -> Tensor:
    # 1. Find max for stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # 2. Subtract max (the shift)
    shifted = x.data - max_vals

    # 3. Compute log-sum-exp safely
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # 4. Result = input - max - log_sum_exp
    return Tensor(x.data - max_vals - log_sum_exp)

class MSELoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
       # 1. Element-wise difference
       diff = predictions.data - targets.data

       # 2. Square the differences
       squared_diff = diff ** 2

       # 3. Mean reduction
       mse = np.mean(squared_diff)

       return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)


class CrossEntropyLoss:
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # 1. Apply stable log_softmax
        log_probs = log_softmax(logits, dim=-1)

        # 2. Select probability of the correct class
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        # Andvanced Indexing : "Lookup Table" style
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # 3. Negative Log Likelyhood
        cross_entropy = -np.mean(selected_log_probs)
        return Tensor(cross_entropy)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

class BinaryCrossEntropyLoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # 1. Clip to prevent log(0) -> NaN
        eps = 1e-7
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # Compute BCE
        term_1 = targets.data * np.log(clamped_preds)
        term_2 = (1 - targets.data) * np.log(1 - clamped_preds)

        binary_cross_entropy = -np.mean(term_1 + term_2)
        return Tensor(binary_cross_entropy)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
