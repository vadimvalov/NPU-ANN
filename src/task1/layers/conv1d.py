import numpy as np
from ..core.layer import Layer

class Conv1D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="valid"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = np.random.randn(out_channels, in_channels, kernel_size) * 0.01
        self.b = np.zeros((out_channels, 1))

    def _pad(self, x):
        if self.padding == "same":
            pad = (self.kernel_size - 1) // 2
            return np.pad(x, ((0, 0), (0, 0), (pad, pad)))
        return x

    def forward(self, x):
        self.x = x
        x = self._pad(x)

        batch_size, _, length = x.shape

        out_length = (length - self.kernel_size) // self.stride + 1

        out = np.zeros((batch_size, self.out_channels, out_length))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + self.kernel_size

                    region = x[b, :, start:end]
                    out[b, oc, i] = np.sum(region * self.W[oc]) + self.b[oc]

        return out

    def backward(self, grad):
        x = self._pad(self.x)

        batch_size, _, length = x.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        dx = np.zeros_like(x)

        out_length = grad.shape[2]

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + self.kernel_size

                    region = x[b, :, start:end]

                    dW[oc] += grad[b, oc, i] * region
                    db[oc] += grad[b, oc, i]

                    dx[b, :, start:end] += grad[b, oc, i] * self.W[oc]

        if self.padding == "same":
            pad = (self.kernel_size - 1) // 2
            dx = dx[:, :, pad:-pad]

        self.dW = dW
        self.db = db

        return dx

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]