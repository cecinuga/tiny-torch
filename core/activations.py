from core.autograd import ReLUBackward, SigmoidBackward, TanhBackward, GELUBackward, SoftmaxBackward
from core.functions import relu, sigmoid, tanh, gelu, softmax
from typing import override
from core.layer import Layer
from core.tensor import Tensor

class ReLU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(relu(x.data))
        res._grad_fn = ReLUBackward(x)
        return res

class Sigmoid(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(sigmoid(x.data))
        res._grad_fn = SigmoidBackward(x)
        return Tensor(res)

class Tanh(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(tanh(x.data))
        res._grad_fn = TanhBackward(x)
        return res

class GELU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(gelu(x.data))
        res._grad_fn = GELUBackward(x)
        return res

class Softmax(Layer):
    @override
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        res = Tensor(softmax(x.data, dim))
        res._grad_fn = SoftmaxBackward(x)
        return res
