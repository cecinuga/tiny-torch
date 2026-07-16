from core.optimizer import SGD
from core.tensor import Tensor
from core.losses import MSELoss
from core.layers import Sequential, Linear
import numpy as np
import matplotlib.pyplot as plt

DATASET_SIZE = 3

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
mean, std = noisy_y.mean(), noisy_y.std()
dataset = Tensor((noisy_y-mean)/std)

# Model architecture
model = Sequential(
    Linear(DATASET_SIZE, DATASET_SIZE),
)
loss = MSELoss()
optimizer = SGD(model.parameters, 1e-4)

# Train loop
for i in range(5000):
    pred = model(dataset)
    train_loss = loss(pred, dataset)
    train_loss.backward()
    optimizer.step()


out = model(dataset)
plt.plot(domain, dataset.data, 'r.')
plt.plot(domain, out.data)
plt.show()
