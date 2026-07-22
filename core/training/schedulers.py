from docutils.parsers.docutils_xml import Unknown
from typing import override
from abc import ABC, abstractmethod
from core.tensor import Tensor
import numpy as np

def clip_grad_norm(parameters: list[Tensor], max_norm: float = 1.0) -> float:
    # 1. Compute global norm across all parameters
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            # Access raw data to avoid graph overhead
            grad_data = param.grad
            total_norm += np.sum(grad_data ** 2)
    total_norm = np.sqrt(total_norm)

    # 2. Scale uniformly if norm exceeds threshold
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_coef

    return float(total_norm)

class Schedule(ABC):
    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        pass

    @abstractmethod
    def get_state(self) -> dict[str, int|float]:
        pass


class CosineSchedule(Schedule):
    def __init__(self, max_lr: float, min_lr: float, total_epochs: int):
        self.max_lr:float = max_lr
        self.min_lr:float = min_lr
        self.total_epochs:int = total_epochs

    @override
    def get_lr(self, epoch: int) -> float:
        # Boundary condition
        if epoch >= self.total_epochs:
            return self.min_lr

        # Cosine annealing formula
        cosine_factor:np.ndarray = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return float(self.min_lr + (self.max_lr - self.min_lr) * cosine_factor)

    @override
    def get_state(self) -> dict[str, int|float]:
        return {'max_lr': self.max_lr, 'min_lr': self.min_lr, 'total_epochs': self.total_epochs}
