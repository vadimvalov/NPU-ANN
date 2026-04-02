import numpy as np
from ..core.layer import Layer


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="valid"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

        kH, kW = self.kernel_size
        self.W = np.random.randn(out_channels, in_channels, kH, kW) * 0.01
        self.b = np.zeros((out_channels, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _pad(self, x):
        if self.padding == "same":
            kH, kW = self.kernel_size
            pH = (kH - 1) // 2
            pW = (kW - 1) // 2
            return np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
        return x

    def forward(self, x):
        self.x = x
        x_padded = self._pad(x)

        batch_size, _, H, W = x_padded.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride

        out_H = (H - kH) // sH + 1
        out_W = (W - kW) // sW + 1

        out = np.zeros((batch_size, self.out_channels, out_H, out_W))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        w_start = j * sW
                        region = x_padded[b, :, h_start:h_start + kH, w_start:w_start + kW]
                        out[b, oc, i, j] = np.sum(region * self.W[oc]) + self.b[oc]

        return out

    def backward(self, grad):
        x_padded = self._pad(self.x)

        batch_size, _, H, W = x_padded.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        out_H, out_W = grad.shape[2], grad.shape[3]

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx_padded = np.zeros_like(x_padded)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        h_start = i * sH
                        w_start = j * sW
                        region = x_padded[b, :, h_start:h_start + kH, w_start:w_start + kW]

                        dW[oc] += grad[b, oc, i, j] * region
                        db[oc] += grad[b, oc, i, j]
                        dx_padded[b, :, h_start:h_start + kH, w_start:w_start + kW] += grad[b, oc, i, j] * self.W[oc]

        if self.padding == "same":
            kH, kW = self.kernel_size
            pH, pW = (kH - 1) // 2, (kW - 1) // 2
            dx = dx_padded[:, :, pH:-pH, pW:-pW] if pH > 0 and pW > 0 else dx_padded
        else:
            dx = dx_padded

        self.dW = dW
        self.db = db

        return dx

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]