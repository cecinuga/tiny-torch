from core.training import Schedule
from core.losses import Loss
from core.optimizer import Optimizer
from core.layers import Layer

class Trainer:
    def __init__(self,
        model: Layer,
        optimizer: Optimizer,
        loss_fn: Loss,
        scheduler: Schedule
    )
