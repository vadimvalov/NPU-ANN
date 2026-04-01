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

from .layers.conv1d import Conv1D

def test_conv1d():
    np.random.seed(42)

    x = np.random.randn(2, 3, 10)  # (batch, channels, length)

    conv = Conv1D(
        in_channels=3,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding="same"
    )

    # forward
    out = conv.forward(x)
    
    print("\n\n====================================\n\n")
    print("FORWARD SHAPE:", out.shape)

    # backward
    grad = np.random.randn(*out.shape)
    dx = conv.backward(grad)

    print("GRAD INPUT SHAPE:", dx.shape)

    print("dW:", conv.dW)
    print("db:", conv.db)

if __name__ == "__main__":
    main()
    test_conv1d()

# for test_conv1d() we see:

# FORWARD SHAPE: (2, 2, 10)
# GRAD INPUT SHAPE: (2, 3, 10)
# dW: [[[-4.8273876  -3.98829166 -1.20567622]
#   [ 3.03422389 -0.56549329 -3.78142209]
#   [-5.32627751 -0.20822592  1.68999698]]

#  [[-1.86677394 -4.30629381  3.41680739]
#   [-1.06102487 -0.12746116 -0.51841376]
#   [ 6.07037585  0.39489516 -2.70212315]]]
# db: [[-1.77179372]
#  [-1.61184475]]

# it means that the output shape is (2, 2, 10)
# and the input shape is (2, 3, 10)
# and grads are not zero, which is good