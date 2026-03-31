import numpy as np

from .layers.linear import Linear
from .activations.relu import ReLU
from .core.sequential import Sequential
from .losses.mse import MSELoss


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)

    # fake target
    y = np.random.randn(2, 4)

    model = Sequential([
        Linear(3, 4),
        ReLU()
    ])

    loss_fn = MSELoss()

    print("INPUT:")
    print(x)

    print("\nTARGET:")
    print(y)

    # forward
    preds = model.forward(x)

    print("\nPREDICTIONS:")
    print(preds)

    # loss
    loss = loss_fn.forward(preds, y)
    print("\nLOSS:")
    print(loss)

if __name__ == "__main__":
    main()

# INPUT:
# [[ 0.49671415 -0.1382643   0.64768854]
#  [ 1.52302986 -0.23415337 -0.23413696]]

# TARGET:
# [[ 1.57921282  0.76743473 -0.46947439  0.54256004]
#  [-0.46341769 -0.46572975  0.24196227 -1.91328024]]

# PREDICTIONS:
# [[-0.         -0.         -0.          0.00259151]
#  [-0.         -0.         -0.          0.00505503]]

# LOSS:
# 0.9706321916512981

# Now we know that we have some problems, that's why we need MSE