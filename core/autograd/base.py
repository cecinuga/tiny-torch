import numpy as np
from core.tensor import Tensor

def enable_autograd(quiet:bool=False) -> None:
    pass

class Function:
    def __init__(self, *tensors: Tensor):
        # The Memory Cost
        self.saved_tensors:tuple[Tensor, ...] = tensors
        # The Graph Structure
        self.next_functions:list[Function|None] = [t._grad_fn for t in tensors]

    def apply(self, grad_output:np.ndarray)-> tuple[np.ndarray, ...]:
        """Compute gradients for the inputs."""
        raise NotImplementedError()
