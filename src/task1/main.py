import numpy as np

from .activations.sigmoid import Sigmoid
from .activations.tanh import Tanh


def main():
    np.random.seed(42)

    x = np.random.randn(2, 3)

    sigmoid = Sigmoid()
    tanh = Tanh()

    print("INPUT:")
    print(x)

    # forward
    out_sigmoid = sigmoid.forward(x)
    out_tanh = tanh.forward(x)

    print("\nSIGMOID OUTPUT:")
    print(out_sigmoid)

    print("\nTANH OUTPUT:")
    print(out_tanh)

    # backward
    grad_out = np.ones_like(x)

    grad_sigmoid = sigmoid.backward(grad_out)
    grad_tanh = tanh.backward(grad_out)

    print("\nSIGMOID GRAD:")
    print(grad_sigmoid)

    print("\nTANH GRAD:")
    print(grad_tanh)


if __name__ == "__main__":
    main()

# INPUT:
# [[ 0.49671415 -0.1382643   0.64768854]
#  [ 1.52302986 -0.23415337 -0.23413696]]

# SIGMOID OUTPUT:
# [[0.62168683 0.46548889 0.65648939]
#  [0.82098421 0.44172766 0.44173171]]

# TANH OUTPUT:
# [[ 0.45952909 -0.13738992  0.57011185]
#  [ 0.90922422 -0.22996582 -0.22995027]]

# SIGMOID GRAD:
# [[0.23519231 0.24880898 0.22551107]
#  [0.14696914 0.24660433 0.24660481]]

# TANH GRAD:
# [[0.78883302 0.98112401 0.67497248]
#  [0.17331133 0.94711572 0.94712287]]

# ok it just works, tanh is literally taken from the numpy
# sigmoid is literally just a 1/(1+e^-x) func