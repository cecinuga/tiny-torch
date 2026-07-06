# %% Setup
import numpy as np
import matplotlib.pyplot as plt
from core.losses import log_softmax
from core.tensor import Tensor

# %% Forward pass
tensor = Tensor([1, 2, 3])
res = log_softmax(tensor)
print(res)
