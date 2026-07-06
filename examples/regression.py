import numpy as np
import matplotlib.pyplot as plt

def rescale(x: np.ndarray, n: float):
    """Rescale the (0, 1) range in (-n, n)"""
    return n*(2*x - 1)

def f(x: np.ndarray):
    """The function we want to estimate"""
    return x*2 + 5

domain = rescale(np.random.random(size=20), 100) # Equivalent of np.linspace(-10, 10, 100)
y = f(domain)

noisy_y = y + np.random.rand(*y.shape)*30

plt.plot(domain, noisy_y, 'r.')
plt.show()
