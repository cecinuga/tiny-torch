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
