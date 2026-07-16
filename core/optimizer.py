from typing import override
import numpy as np
from core.tensor import Tensor

class Optimizer:
    def __init__(self, params: list[Tensor]):
        self.params:list[Tensor] = params
        self.step_count:int = 0

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self) -> None:
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.01, weight_decay=0.0):
        super().__init__(params)
        self.lr:float = lr
        self.weight_decay = weight_decay

    @override
    def step(self) -> None:
        for param in self.params:
            if param.grad is None: continue

            grad_data = param.data
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.grad

            param.data = param.data - self.lr * grad_data
        self.step_count += 1

class SGDM(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.01, momentum: float=0.0, weight_decay=0.0):
        super().__init__(params)
        self.momentum_buffer:list[np.ndarray|None] = [None for _ in params]
        self.weight_decay:float = weight_decay
        self.momentum:float = momentum
        self.lr:float = lr

    @override
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None: continue
            grad_data = param.grad
            if self.momentum != 0:
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = np.zeros_like(param.data)

                # Update velocity: v = momentum * v_prev + grad
                self.momentum_buffer[i] = (self.momentum * self.momentum_buffer[i]) + grad_data  # ty:ignore[unsupported-operator]  # pyright: ignore[reportOperatorIssue]
                grad_data = self.momentum_buffer[i]

            param.data = param.data - (self.lr * grad_data)  # ty:ignore[unsupported-operator]  # pyright: ignore[reportOperatorIssue]

    def has_momentum(self) -> bool:
        return self.momentum > 0

    def get_momentum_state(self) -> list[np.ndarray|None]:
        return self.momentum_buffer

    def set_momentum_state(self, buffers: list[np.ndarray|None]) -> None:
        self.momentum_buffer = buffers

class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr: float=0.001, betas:tuple[float, float]=(0.9, 0.999), eps: float=1e-8, weight_decay:float=0.0):
        super().__init__(params)
        self.lr:float = lr
        self.eps:float = eps
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m_buffers:list[np.ndarray|None] = [None for _ in params]
        self.v_buffers:list[np.ndarray|None] = [None for _ in params]

    @override
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None: continue
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.grad)
            if self.v_buffers[i] is None:
                self.v_buffers[i] = np.zeros_like(param.grad)

            # 1. Update biased moments
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * param.grad  # ty:ignore[unsupported-operator]  # pyright: ignore[reportOperatorIssue]
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (param.grad**2)  # ty:ignore[unsupported-operator]  # pyright: ignore[reportOperatorIssue]

            # 2. Compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            # 3. Update parameter (Adaptive step)
            param.data = param.data - (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)

class AdamW(Adam):
    def __init__(self, params: list[Tensor], lr: float, eps: float, beta1: float, beta2: float, weight_decay: float):
        super().__init__(params, lr, eps, beta1, beta2)
        self.params:list[Tensor] = params
        self.weight_decay:float = weight_decay
        self.lr:float = lr


    @override
    def step(self) -> None:
        super().step()
        for param in self.params:
            param.data = param.data * (1 - self.lr * self.weight_decay)
