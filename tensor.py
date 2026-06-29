from typing import Any
import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.shape = data.shape

    def __add__(self, other: Tensor | np.ndarray):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)

    def matmul(self, other: Tensor) -> Tensor:
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-1]:
                raise ValueError(f"cannot perform matrix multiplication: {self.shape} @ {other.shape}\ninner dimension must match: {self.shape[-1]} != {other.shape[-1]}")

        result_data = np.matmul(self.data, other.data)
        return Tensor(result_data)
