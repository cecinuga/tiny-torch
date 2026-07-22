from core.tensor import Tensor
import pickle
from pathlib import Path
import numpy as np

from core.dataset import DataLoader
from core.training import Schedule, clip_grad_norm
from core.losses import Loss
from core.optimizer import Optimizer
from core.layers import Layer

class Trainer:
    def __init__(self,
        model: Layer,
        optimizer: Optimizer,
        loss_fn: Loss,
        scheduler: Schedule|None,
        grad_clip_norm: float|None = None
    ):
        self.model: Layer = model
        self.optimizer: Optimizer = optimizer
        self.loss_fn: Loss = loss_fn
        self.scheduler: Schedule|None = scheduler
        self.grad_clip_norm: float|None = grad_clip_norm

        # State tracking
        self.epoch: int = 0
        self.training: bool = True
        self.history:dict[str, list[float]] = {'train_loss': [], 'eval_loss': []}

    def train_epoch(self, dataloader: DataLoader, accumulation_step:int = 1):
        self.model.train()
        accumulated_loss: float = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # 1. Forward pass
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)

            # 2. Scale loss for accumulation
            # Dividing by N so the sum of N gradients equals to mean
            scaled_loss = loss.data / accumulation_step
            accumulated_loss += float(scaled_loss)

            # 3. Backward pass (accumulates into .grad)
            loss.backward()

            # Only update every 'accumulation_step'
            if (batch_idx + 1) % accumulation_step == 0:
                # 4. Gradient Clipping (Safety)
                if self.grad_clip_norm is not None:
                    _ = clip_grad_norm(self.model.parameters, self.grad_clip_norm)

                self.history['train_loss'].append(accumulated_loss)
                # 5. Optimizer Step (Update)
                self.optimizer.step()
                self.optimizer.zero_grad() # Clear buffers

            if self.scheduler is not None:
                self.optimizer.lr = self.scheduler.get_lr(self.epoch)

            self.epoch += 1

    def eval(self, dataloader: DataLoader) -> float:
        self.model.eval()
        self.training = False

        total_loss:float = 0.0
        for inputs, targets in dataloader:
            # Forward pass only
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets)

            total_loss += float(loss.data)

        self.model.train()
        self.training = True
        return total_loss / len(dataloader)

    def _get_model_state(self) -> list[Tensor]:
        return self.model.parameters

    def _get_optimizer_state(self):
        return self.optimizer.get_state()

    def _get_scheduler_state(self):
        if self.scheduler is not None:
            return self.scheduler.get_state()
        return None

    def save(self, path: Path|str) -> None:
        checkpoint = {
            'epoch':            self.epoch,
            'model_state':      self._get_model_state(),
            'optimizer_state':  self._get_optimizer_state(),
            'scheduler_state':  self._get_scheduler_state(),
            'history':          self.history
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
