import numpy as np

class Tensor:
    def __init__(self, data):
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = np.stack([t.data for t in data])
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor=(shape={self.shape}, size={self.size}, dtype={self.dtype})"

    def __str__(self):
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def __add__(self, other: Tensor | np.ndarray):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)

    def __sub__(self, other: Tensor | np.ndarray):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)

    def __matmul__(self, other: Tensor | np.ndarray):
        return self.matmul(other)

    def __truediv__(self, other:Tensor | np.ndarray):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)

    def matmul(self, other: Tensor | np.ndarray) -> Tensor:
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"cannot perform matrix multiplication: {self.shape} @ {other.shape}\n"+
                    f"inner dimension must match: {self.shape[-1]} != {other.shape[-2]}"
                )

        result_data = np.matmul(self.data, other.data)
        return Tensor(result_data)

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in shape:
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"cannot reshape {self.shape} to {new_shape}\n"+
                f"[x] Element count mismatch: {self.size} elements vs {target_size} elements\n"+
                f"[x] Reshape preserves data, so total elements must stay the same\n"+
                f"[x] Use -1 to infer a dimension: reshape(-1, {new_shape[-1] if len(new_shape) > 0 else 1}) lets NumPy calculate"
            )

        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)

    def transpose(self):
        new_shape = list(a.shape)
        new_shape.reverse()
        return self.reshape(new_shape)

    def sum(self, axis=None, keepdims=False) -> Tensor:
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False) -> Tensor:
       return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False) -> Tensor:
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False) -> Tensor:
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims))
