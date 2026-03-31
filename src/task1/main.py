import numpy as np

from .layers.linear import Linear
from .activations.relu import ReLU


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)

    layer = Linear(3, 4)
    relu = ReLU()

    print("INPUT:")
    print(x)

    # forward
    out = layer.forward(x)
    out_relu = relu.forward(out)

    print("\nOUTPUT:")
    print(out)

    print("\nOUTPUT RELU:")
    print(out_relu)

    # backward
    grad_out = np.ones_like(out_relu)

    grad_relu = relu.backward(grad_out)
    grad_input = layer.backward(grad_relu)

    print("\nGRAD AFTER RELU:")
    print(grad_relu)

    print("\nGRAD INPUT:")
    print(grad_input)

    print("\nGRAD W:")
    print(layer.dW)

    print("\nGRAD b:")
    print(layer.db)


if __name__ == "__main__":
    main()


# INPUT:
# [[ 0.49671415 -0.1382643   0.64768854]
#  [ 1.52302986 -0.23415337 -0.23413696]]

# OUTPUT:
# [[-0.00268718  0.00081402 -0.00922648  0.0073757 ]
#  [ 0.02917566  0.0140953  -0.00534539  0.01200759]]

# OUTPUT RELU:
# [[-0.          0.00081402 -0.          0.0073757 ]
#  [ 0.02917566  0.0140953  -0.          0.01200759]]

# GRAD AFTER RELU:
# [[0. 1. 0. 1.]
#  [1. 1. 0. 1.]]

# GRAD INPUT:
# [[ 0.01309995 -0.0237901  -0.0024804 ]
#  [ 0.02889208 -0.02842428 -0.01972958]]

# GRAD W:
# [[ 1.52302986  2.01974401  0.          2.01974401]
#  [-0.23415337 -0.37241768  0.         -0.37241768]
#  [-0.23413696  0.41355158  0.          0.41355158]]

# GRAD b:
# [[1. 2. 0. 2.]]

# we have 2x3 as input, 2x4 as output
# grad after relu is a mask of positive elements in OUTPUT (since grad_out is all ones)
# grad b = sum(grad after relu, axis=0) = [0+1, 1+1, 0+0, 1+1] = [1, 2, 0, 2]
# grad W = INPUT.T @ grad after relu (e.g., column 3 of GRAD W is 0 since column 3 of GRAD AFTER RELU is all zeros)
# grad_input = grad after relu @ W.T

# and ofc 1. means 1.0 