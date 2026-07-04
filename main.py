from core.activations import GELU
from core.tensor import Tensor
from core.losses import log_softmax

def main():
    a = Tensor([[1,2,3], [4,5,6]])
    print(log_softmax(a))


if __name__ == "__main__":
    main()
