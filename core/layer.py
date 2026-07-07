from typing import override, Any
import numpy as np
from core.tensor import Tensor

class Layer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def __name__(self) -> str:
        return "Layer"

    def __repr__(self) -> str:
        return f"{self.__name__}=({self.args}, {self.kwargs})"

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Compute layer output"""
        raise NotImplementedError

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(x, *args, **kwargs)

    @property
    def parameters(self) -> list[Tensor]:
        """Return list of trainable parameters"""
        return []

class Linear(Layer):
    def __init__(self, in_feature:int, out_feature:int, bias:bool=True):
        super().__init__(in_feature, out_feature, bias)
        self.in_feature = in_feature

        # Xavier initialization
        scale = np.sqrt(1.0 / in_feature)
        weight_data = np.random.randn(in_feature, out_feature) * scale
        self.weight = Tensor(weight_data)

        if bias:
            self.bias = Tensor(np.zeros(out_feature))
        else:
            self.bias = None

    @property
    @override
    def __name__(self):
        return "Linear"

    @override
    def forward(self, x: Tensor, **_):
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
    def __init__(self, p=0.5):
        super().__init__(p=0.5)
        self.p = p

    @override
    def forward(self, x: Tensor, training:bool=True, **_) -> Tensor:
        if not training:
            return x

        # 1. Create Mask
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        # 2. Scale Factor (Inverted Dropout)
        scale = 1.0 / keep_prob

        # 3. Apply
        return x * Tensor(mask) * Tensor(scale)

    @property
    @override
    def __name__(self):
        return "Dropout"

class Sequential(Layer):
    def __init__(self, *layers: Layer):
        super().__init__(layers)
        self.layers: list[Layer] = list(layers)

    @override
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x, **kwargs)
            except Exception as e:
                raise Exception(f"layer {i}: {layer}", e)
        return x

    @property
    @override
    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params
