from core.tensor import Tensor
from core.losses import MSELoss
from core.activations import ReLU
from core.layers import Sequential, Linear
import numpy as np
import matplotlib.pyplot as plt

DATASET_SIZE = 10

# Utils
def rescale(x: np.ndarray, n: float):
    """Rescale the (0, 1) range in (-n, n)"""
    return n*(2*x - 1)

# To estimate function
def f(x: np.ndarray):
    """The function we want to estimate"""
    return x*2 + 5

# Datase
domain = rescale(np.random.random(size=DATASET_SIZE), 5) # Equivalent of np.linspace(-10, 10, 100)
y = f(domain)
noisy_y = y + np.random.rand(*y.shape)*30
dataset = Tensor(noisy_y)

# Model architecture
model = Sequential(
    Linear(DATASET_SIZE, DATASET_SIZE),
)
loss = MSELoss()

# Train loop
for i in range(20):
    pred = model(dataset)
    print(pred)
    train_loss = loss(pred, dataset)
    print(train_loss)
    exit(1)

plt.plot(domain, noisy_y, 'r.')
plt.show()
