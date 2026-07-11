from core.functions import relu, sigmoid, tanh, gelu, softmax
from typing import override
from core.layer import Layer
from core.tensor import Tensor

class ReLU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU: max(0, x)"""
        res = relu(x.data)
        return Tensor(res)

class Sigmoid(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = sigmoid(x.data)
        return Tensor(res)

class Tanh(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = tanh(x.data)
        return Tensor(res)

class GELU(Layer):
    @override
    def forward(self, x: Tensor) -> Tensor:
        res = gelu(x.data)
        return Tensor(res)

class Softmax(Layer):
    @override
    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        res = softmax(x.data, dim)
        return Tensor(res)
