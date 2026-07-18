from core.autograd import CrossEntropyLossBackward, MSELossBackward, BCELossBackward
from core.functions import mse, cross_entropy, binary_cross_entropy
from core.tensor import Tensor

class MSELoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(mse(predictions.data, targets.data))
        out._grad_fn = MSELossBackward(predictions, targets)
        return out

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

class CrossEntropyLoss:
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(cross_entropy(logits.data, targets.data))
        out._grad_fn = CrossEntropyLossBackward(logits, targets)

        return out

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

class BinaryCrossEntropyLoss:
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        out = Tensor(binary_cross_entropy(predictions.data, targets.data))
        out._grad_fn = BCELossBackward(predictions, targets)

        return out

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
