from abc import ABC, abstractmethod
from typing import override
import numpy as np
from core.tensor import Tensor

class Layer(ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute layer output"""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def parameters(self) -> list[Tensor]:
        return []

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

class Linear(Layer):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True):
        self.in_feature = in_feature
        self.out_feature = out_feature

        # Xavier initialization
        scale = np.sqrt(1.0 / in_feature)
        weight_data = np.random.randn(in_feature, out_feature) * scale
        self.weight = Tensor(weight_data)

        if bias:
            self.bias = Tensor(np.zeros(out_feature))
        else:
            self.bias = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, out_feature={self.out_feature}, bias={self.bias})"

    @override
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Compute the layer output: y = xW + b"""
        # 1. Matrix multiplication
        output = x.matmul(self.weight)

        # 2. Bias Addition (Broadcasting)
        if self.bias is not None:
            output = output + self.bias

        return output

    @property
    @override
    def parameters(self):
        """Return the list of trainable parameters in the layer"""
        params = [self.weight]

        if self.bias is not None:
            params.append(self.bias)

        return params

class Dropout(Layer):
    def __init__(self, p:float=0.5):
        self.p:float = p

    @override
    def forward(self, x: Tensor, training:bool=True, *_, **__) -> Tensor:
        if not training:
            return x

        # 1. Create Mask
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        # 2. Scale Factor (Inverted Dropout)
        scale = 1.0 / keep_prob

        # 3. Apply
        return x * Tensor(mask) * Tensor(scale)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers: list[Layer] = list(layers)

    @override
    def __repr__(self) -> str:
        return f"Sequential=({[l for l in self.layers]})"

    @override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    @override
    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params
