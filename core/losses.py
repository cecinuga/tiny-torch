from core.functions import mse, cross_entropy, binary_cross_entropy
import numpy as np
from core.tensor import Tensor

class MSELoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        err = mse(predictions.data, targets.data)
        return Tensor(err)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

class CrossEntropyLoss:
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        res = cross_entropy(logits.data, targets.data)
        return Tensor(res)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

class BinaryCrossEntropyLoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        res = binary_cross_entropy(predictions.data, targets.data)
        return Tensor(res)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
