import numpy as np

from .layers.conv1d import Conv1D
from .core.sequential import Sequential
from .losses.mse import MSELoss
from .optimizers.sgd import SGD


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3, 10)
    y = np.random.randn(2, 2, 10)

    model = Sequential([
        Conv1D(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding="same"
        ),
    ])

    loss_fn = MSELoss()
    optimizer = SGD(model.layers, lr=0.01)

    for epoch in range(101):

        # forward
        preds = model.forward(x)
        loss = loss_fn.forward(preds, y)

        # backward
        grad = loss_fn.backward()
        model.backward(grad)

        # update
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    print("\nFinal output shape:", preds.shape)


if __name__ == "__main__":
    main()