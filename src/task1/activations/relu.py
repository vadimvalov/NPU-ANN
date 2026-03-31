from ..core.layer import Layer

class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask