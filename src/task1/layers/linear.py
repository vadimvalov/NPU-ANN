import numpy as np
from ..core.layer import Layer

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]