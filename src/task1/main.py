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
        # ReLU()
    ])

    loss_fn = MSELoss()
    optimizer = SGD(model.layers, lr=0.01)

    # forward
    preds = model.forward(x)
    loss = loss_fn.forward(preds, y)

    print("LOSS BEFORE:")
    print(loss)

    # backward
    grad = loss_fn.backward()
    model.backward(grad)

    linear_layer = model.layers[0]
    old_W = linear_layer.W.copy()

    optimizer.step()

    print("\nWEIGHTS BEFORE:")
    print(old_W)

    print("\nWEIGHTS AFTER:")
    print(linear_layer.W)

    print("\nWEIGHT CHANGE:")
    print(linear_layer.W - old_W)

    print("\nLOSS AFTER:")
    print(loss_fn.forward(model.forward(x), y))

if __name__ == "__main__":
    main()

# LOSS BEFORE:
# 0.9722058667327826

# WEIGHTS BEFORE:
# [[-0.01724918 -0.00562288 -0.01012831  0.00314247]
#  [-0.00908024 -0.01412304  0.01465649 -0.00225776]
#  [ 0.00067528 -0.01424748 -0.00544383  0.00110923]]

# WEIGHTS AFTER:
# [[-0.01695156 -0.00642337 -0.00970992 -0.00349121]
#  [-0.00937144 -0.01412028  0.01466317 -0.00132145]
#  [ 0.00350054 -0.01271705 -0.0063388   0.00310644]]

# WEIGHT CHANGE:
# [[ 2.97618598e-04 -8.00494745e-04  4.18393632e-04 -6.63367973e-03]
#  [-2.91197836e-04  2.75372795e-06  6.68716143e-06  9.36315768e-04]
#  [ 2.82525982e-03  1.53043624e-03 -8.94969166e-04  1.99721153e-03]]

# LOSS AFTER:
# 0.9640448544347593

# quite a tricky one tho, we've learned NN to learn.
# weights are changing
# tho we need to comment relu activation here cuz it is falling down
# due to negative values

# loss goes down mood goes up