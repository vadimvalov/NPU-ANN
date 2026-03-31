import numpy as np

from .layers.linear import Linear
from .activations.relu import ReLU
from .core.sequential import Sequential


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)

    model = Sequential([
        Linear(3, 4),
        ReLU()
    ])

    print("INPUT:")
    print(x)

    # forward
    out = model.forward(x)

    print("\nOUTPUT:")
    print(out)

    # backward
    grad_out = np.ones_like(out)
    grad_input = model.backward(grad_out)

    print("\nGRAD INPUT:")
    print(grad_input)

    linear_layer = model.layers[0] # so we can check grads further

    print("\nGRAD W:")
    print(linear_layer.dW)

    print("\nGRAD b:")
    print(linear_layer.db)


if __name__ == "__main__":
    main()

# INPUT:
# [[ 0.49671415 -0.1382643   0.64768854]
#  [ 1.52302986 -0.23415337 -0.23413696]]

# OUTPUT:
# [[-0.          0.00081402 -0.          0.0073757 ]
#  [ 0.02917566  0.0140953  -0.          0.01200759]]

# GRAD INPUT:
# [[ 0.01309995 -0.0237901  -0.0024804 ]
#  [ 0.02889208 -0.02842428 -0.01972958]]

# GRAD W:
# [[ 1.52302986  2.01974401  0.          2.01974401]
#  [-0.23415337 -0.37241768  0.         -0.37241768]
#  [-0.23413696  0.41355158  0.          0.41355158]]

# GRAD b:
# [[1. 2. 0. 2.]]

# Sequential applies layers one by one:
# forward: Linear → ReLU
# backward: ReLU → Linear (reverse order)
# Output shape: 2x4 (same as before)
# grad_out = ones → propagated through ReLU mask
# backward works automatically:
# Sequential handles chaining of gradients between layers

# tl;dr 
# result stays the same, proves it works well lol
# but now we wrapped it into sequential model, taking the layer and activation