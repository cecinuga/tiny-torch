import numpy as np
from core.tensor import Tensor
from core.layer import Linear

def main():
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    new = np.broadcast_to(x, (1, 3))

    print(x)
    print(new)

if __name__ == "__main__":
    main()
