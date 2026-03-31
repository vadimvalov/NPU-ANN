import numpy as np

from .layers.linear import Linear


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)

    layer = Linear(3, 4)

    print("INPUT:")
    print(x)

    # forward
    out = layer.forward(x)

    print("\nOUTPUT:")
    print(out)

    # fake gradient
    grad_out = np.ones_like(out)

    # backward
    grad_input = layer.backward(grad_out)

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

# GRAD INPUT:
# [[ 0.02419733 -0.02600465 -0.02985789]
#  [ 0.02419733 -0.02600465 -0.02985789]]

# GRAD W:
# [[ 2.01974401  2.01974401  2.01974401  2.01974401]
#  [-0.37241768 -0.37241768 -0.37241768 -0.37241768]
#  [ 0.41355158  0.41355158  0.41355158  0.41355158]]

# GRAD b:
# [[2. 2. 2. 2.]]

# we have 2x3 as input, 2x4 as output
# [1+1, 1+1, 1+1, 1+1] = [2,2,2,2] - this is grad b
# grad_input = grad_out @ W.T