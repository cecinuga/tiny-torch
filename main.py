from core.activations import GELU
from core.tensor import Tensor

def main():
    a = Tensor([[[1,2,3], [4,5,6], [7, 8, 9], [10, 11, 12]], [[13,14,15], [16,17,18], [19, 20, 21], [22, 23, 24]]])
    gelu = GELU()

    print(a.shape)
    print(gelu(a))
    print()


if __name__ == "__main__":
    main()
