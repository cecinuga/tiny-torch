from core.activations import ReLU
from core.optimizer import SGD
from core.tensor import Tensor
from core.losses import MSELoss
from core.layers import Sequential, Linear
from core.loader import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Configuration
EPOCHS = 20
TEST_STEP = 5
DATASET_SIZE = 20
NOISY = 2


# To estimate function
def f(x: np.ndarray):
    """The function we want to estimate"""
    return x*2 + 5


# Dataset
domain = np.linspace(-1, 1, DATASET_SIZE)
y = f(domain)
train_data = Tensor(y + np.random.rand(*y.shape)*NOISY)
test_data = Tensor(y + np.random.rand(*y.shape)*NOISY)
dataset = TensorDataset(train_data, test_data)


# Model architecture
model = Sequential(
    Linear(DATASET_SIZE, DATASET_SIZE),
    ReLU(),
)
model.save_graph("./tmp/architecture.png", arch=True)
model.save_graph("./tmp/forward.png", arch=False, forward=True)
model.save_graph("./tmp/backward.png", arch=False, backward=True)

loss = MSELoss()
optimizer = SGD(model.parameters, 1e-4)

# Train loop
test_losses: list[np.ndarray] = []
train_losses: list[np.ndarray] = []
for i in range(EPOCHS):
    pred = model(dataset.tensors[0])
    train_loss = loss(pred, dataset.tensors[0])
    if i%TEST_STEP == 0:
        model.eval()
        test_pred = model(dataset.tensors[1])
        test_loss = loss(test_pred, dataset.tensors[1])
        test_losses.append(test_loss.data)
        model.train()

    train_losses.append(train_loss.data)

    train_loss.backward()
    optimizer.step()

# Eval
out = model(dataset.tensors[1])

# Plot in one window, divided into sections
fig, (ax_loss, ax_result) = plt.subplots(1, 2, figsize=(12, 5))

# Loss section
ax_loss.plot(list(range(EPOCHS)), train_losses, label="train")
ax_loss.plot(list(range(0, EPOCHS, TEST_STEP)), test_losses, label="test")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss")
ax_loss.legend()

# Result section (the regression)
ax_result.plot(domain, dataset.tensors[1].data, 'r.', label="target")
ax_result.plot(domain, out.data, label="prediction")
ax_result.set_title("Regression")
ax_result.set_xlabel("x")
ax_result.set_ylabel("y")
ax_result.legend()

fig.tight_layout()
plt.show()
