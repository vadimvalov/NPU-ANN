import numpy as np

from .layers.linear import Linear
from .activations.relu import ReLU
from .core.sequential import Sequential
from .losses.mse import MSELoss
from .optimizers.sgd import SGD


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)
    y = np.random.randn(2, 4)

    model = Sequential([
        Linear(3, 4),
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


if __name__ == "__main__":
    main()

# Epoch 0, Loss: 0.9722058667327826
# Epoch 10, Loss: 0.8962557009499458
# Epoch 20, Loss: 0.831358903212809
# Epoch 30, Loss: 0.7754564755383733
# Epoch 40, Loss: 0.7268942292872348
# Epoch 50, Loss: 0.6843424132369282
# Epoch 60, Loss: 0.6467313263264196
# Epoch 70, Loss: 0.6131997350314083
# Epoch 80, Loss: 0.5830535480100252
# Epoch 90, Loss: 0.5557327074634683
# Epoch 100, Loss: 0.5307846626296182

# it learns, it comes closer and closer to zero with 101 epochs
# it is smart, almost smarter than me atp