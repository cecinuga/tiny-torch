class Layer:
    def forward(self, x):
        """Compute the layer output"""
        raise NotImplementedError("Forward method not implemented.")

    def parameters(self):
        """Return the list of trainable parameters in the layer"""
        return []

    def __call__(self, x, *args, **kwargs):
        """Allow the layer to be called like a function"""
        return self.forward(x, *args, **kwargs)