import numpy as np
from core.tensor import Tensor

class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU: max(0, x)
        Cost: 1x (Baseline)
        """
        return Tensor(np.maximum(0, x.data))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Sigmoid:
    def forward(self, x: Tensor) -> Tensor:
        # 1. Clip to prevent raw overflow
        z = np.clip(x.data, -500, 500)

        # 2. Stable computation mask
        result = np.zeros_like(z)

        # Positive input: standard formula
        pos_mask = z >= 0
        result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        # Negative input: alternative formula
        neg_mask = z <= 0
        exp_z = np.exp(z[neg_mask])
        result[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Tanh:
    def forward(self, x: Tensor) -> Tensor:
        # Relies on Numpy's internal optimization
        return Tensor(np.tanh(x.data))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class GELU:
    def forward(self, x: Tensor) -> Tensor:
        # Approximation: x + sigmoid(1.702 * x)
        # 1.702 is derived from sqrt(2/pi)
        return Sigmoid()(x * 1.702) * x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Softmax:
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        # 1. Shift values so max is 0
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max

        # 2. Compute safe exponentials
        exp_values = np.exp(x_shifted.data)

        # 3. Normalize
        exp_sum = np.sum(exp_values.data, axis=dim, keepdims=True)
        return Tensor(exp_values / exp_sum)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
