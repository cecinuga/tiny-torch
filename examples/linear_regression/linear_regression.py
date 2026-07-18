from core.optimizer import SGD
from core.tensor import Tensor
from core.losses import MSELoss
from core.layers import Sequential, Linear
import numpy as np
import matplotlib.pyplot as plt

# Configuration
EPOCHS = 100
TEST_STEP = 5
DATASET_SIZE = 50
NOISY = 8


# To estimate function
def f(x: np.ndarray):
    """The function we want to estimate"""
    return x*2 + 5


# Dataset
train_domain = np.linspace(-5, 5, DATASET_SIZE).reshape(-1, 1)
test_domain = np.linspace(-10, 10, DATASET_SIZE).reshape(-1, 1)
y_train = f(train_domain)
y_test = f(test_domain)
train_y = Tensor(y_train + np.random.uniform(-1, 1, y_train.shape)*NOISY)
train_x = Tensor(train_domain)
test_y = Tensor(y_test + np.random.uniform(-1, 1, y_test.shape)*NOISY)
test_x = Tensor(test_domain)


# Testing with closed form
# Design matrix [x, 1] so lstsq fits both slope (m) and intercept (c)
A = np.hstack([test_x.data, np.ones_like(test_x.data)])
m, c = np.linalg.lstsq(A, test_y.data.ravel(), rcond=None)[0]
closed_form = test_x.data*m + c

# Model architecture
model = Sequential(
    Linear(1, 1),   # one slope, one intercept
)
loss = MSELoss()
optimizer = SGD(model.parameters, 1e-2)


# Save graph of models
#model.save_graph("./examples/linear_regression/arch/architecture.png", arch=True)
#model.save_graph("./examples/linear_regression/arch/forward.png", arch=False, forward=True)
#model.save_graph("./examples/linear_regression/arch/backward.png", arch=False, backward=True)


# Train loop
test_losses: list[np.ndarray] = []
train_losses: list[np.ndarray] = []

for i in range(EPOCHS):
    pred = model(train_x)
    train_loss = loss(pred, train_y)
    train_losses.append(train_loss.data)
    if i%TEST_STEP == 0 or i == EPOCHS-1:
        model.eval()
        test_pred = model(test_x)
        test_loss = loss(test_pred, test_y)
        test_losses.append(test_loss.data)
        model.train()

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Eval
out = model(test_x)

# Plot in one window, divided into sections
fig, (ax_loss, ax_result) = plt.subplots(1, 2, figsize=(12, 5))

# Loss section
ax_loss.plot(list(range(EPOCHS)), train_losses, label="train")
ax_loss.plot(list(range(0, EPOCHS+1, TEST_STEP)), test_losses, label="test")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("loss")
ax_loss.legend()

# Result section (the regression)
ax_result.plot(test_x.data, test_y.data, 'r.', label="target")
ax_result.plot(test_x.data, out.data, label="prediction")
ax_result.plot(test_x.data, y_test, label='to estimate')
ax_result.plot(test_x.data, closed_form, label='closed form')
ax_result.set_title("Regression")
ax_result.set_xlabel("x")
ax_result.set_ylabel("y")
ax_result.legend()

fig.tight_layout()
plt.show()
