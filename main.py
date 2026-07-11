from core.tensor import Tensor
from core.losses import MSELoss

def main():
    targets = Tensor([1, 2, 3])
    loss_fn = MSELoss()

    a = Tensor([5, 1, 2], requires_grad=True)
    b = Tensor([3, 4, 1], requires_grad=True)
    c = a * b

    loss = loss_fn(c, targets)

    grad = loss.backward()
    print(grad)

if __name__ == "__main__":
    main()
