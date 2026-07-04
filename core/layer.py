import numpy as np
from core.tensor import Tensor
class Layer:
    def __init__(self, in_feature:int, out_feature:int, bias=True):
        self.in_feature = in_feature
        
        # Xavier initialization
        scale = np.sqrt(1.0 / in_feature)
        weight_data = np.random.randn(in_feature, out_feature) * scale
        self.weight = Tensor(weight_data)

        if bias:
            self.bias = Tensor(np.zeros(out_feature))
        else:
            self.bias = None
    
    def __call__(self, x, *args, **kwargs):
        """Allow the layer to be called like a function"""
        return self.forward(x, *args, **kwargs)

    def forward(self, x: Tensor):
        """Compute the layer output: y = xW + b"""
        # 1. Matrix multiplication
        output = x.matmul(self.weight)

        # 2. Bias Addition (Broadcasting)
        if self.bias is not None:
            output = output + self.bias

        return output
        
    @property
    def parameters(self):
        """Return the list of trainable parameters in the layer"""
        params = [self.weight]
        
        if self.bias is not None:
            params.append(self.bias)

        return params
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, training=True):
        if not training: 
            return x

        # 1. Create Mask
        keep_prob = 1.0 - self.p
        mask = np.random.random(x.data.shape) < keep_prob

        # 2. Scale Factor (Inverted Dropout)
        scale = 1.0 / keep_prob

        # 3. Apply
        return x * Tensor(mask) * Tensor(scale)
    
class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
    
    @property
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

