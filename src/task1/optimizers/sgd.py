class SGD:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            for param, grad in zip(layer.parameters(), layer.gradients()):
                param[:] = param - self.lr * grad