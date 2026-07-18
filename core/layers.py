from abc import ABC, abstractmethod
from pathlib import Path
from typing import override, TYPE_CHECKING
import numpy as np
from core.tensor import Tensor

if TYPE_CHECKING:
    from core.graph import ComputationalGraph

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
        weight_data:np.ndarray = np.random.randn(in_feature, out_feature) * scale
        self.weight:Tensor = Tensor(weight_data)
        # Semantic role used when drawing the computational graph.
        self.weight.role = "weights"

        if bias:
            self.bias = Tensor(np.zeros(out_feature))
            self.bias.role = "bias"
        else:
            self.bias = None

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(in_feature={self.in_feature}, out_feature={self.out_feature}, bias={self.bias})"

    @override
    def forward(self, x: Tensor) -> Tensor:
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
    def forward(self, x: Tensor, training:bool=True) -> Tensor:
        if not training or self.p == 0.0:
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
        self._graph: ComputationalGraph | None = None

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

    def build_graph(self, arch: bool = True, forward: bool = False,
                    backward: bool = False) -> None:
        """Build the graph of this model.

        Args:
            arch: include the network architecture (layer info).
            forward: include the forward computational graph (data flow).
            backward: include the backward computational graph (autograd ops).
        """
        # Imported lazily to avoid a circular import (core.graph imports layers).
        from core.graph import ComputationalGraph
        self._graph = ComputationalGraph(self)
        self._graph.build(arch=arch, forward=forward, backward=backward)

    def save_graph(self, path: str | Path, arch: bool = True, forward: bool = False,
                   backward: bool = False) -> None:
        """Render the graph to a .png image at ``path``."""
        self.build_graph(arch, forward, backward)
        assert self._graph is not None
        self._graph.render(path)

    def destroy_graph(self) -> None:
        self._graph = None
