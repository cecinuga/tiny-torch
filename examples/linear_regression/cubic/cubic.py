from core.optimizer import SGD
from core.tensor import Tensor
from core.losses import MSELoss
from core.layers import Sequential, Linear
import numpy as np
import matplotlib.pyplot as plt

# Configuration
EPOCHS = 50
TEST_STEP = 5
DATASET_SIZE = 50
NOISY = 80


# To estimate function
def f(x: np.ndarray):
    """The function we want to estimate"""
    return 1.2*x**3 - 2.3*x**2 + x*2 + 2


# Dataset
train_domain = np.linspace(-5, 5, DATASET_SIZE).reshape(-1, 1)
test_domain = np.linspace(-10, 10, DATASET_SIZE).reshape(-1, 1)
f_train = f(train_domain)
f_test = f(test_domain)
X_train = Tensor(np.hstack([train_domain, train_domain**2, train_domain**3]))
X_test = Tensor(np.hstack([test_domain, test_domain**2, test_domain**3]))
train_y = Tensor(f_train + np.random.uniform(-1, 1, f_train.shape)*NOISY)
test_y = Tensor(f_test + np.random.uniform(-1, 1, f_test.shape)*NOISY)


# Model architecture
model = Sequential(
    Linear(3, 1),
)
loss = MSELoss()
optimizer = SGD(model.parameters, 1e-4)


# Save graph of models
model.save_graph("./examples/linear_regression/quadratic/arch/architecture.png", arch=True)
model.save_graph("./examples/linear_regression/quadratic/arch/forward.png", arch=False, forward=True)
model.save_graph("./examples/linear_regression/quadratic/arch/backward.png", arch=False, backward=True)


# Train loop
test_losses: list[np.ndarray] = []
train_losses: list[np.ndarray] = []

for i in range(EPOCHS):
    pred = model(X_train)
    train_loss = loss(pred, train_y)
    train_losses.append(train_loss.data)

    if i%TEST_STEP == 0 or i == EPOCHS-1:
        model.eval()
        test_pred = model(X_test)
        test_loss = loss(test_pred, test_y)
        test_losses.append(test_loss.data)
        model.train()

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# Eval
model.eval()
out = model(X_test)


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
ax_result.plot(test_domain, test_y.data, 'r.', label="target")
ax_result.plot(test_domain, out.data, label="prediction")
ax_result.plot(test_domain, f_test, label='to estimate')
ax_result.set_title("Regression")
ax_result.set_xlabel("x")
ax_result.set_ylabel("y")
ax_result.legend()

fig.tight_layout()
plt.show()
