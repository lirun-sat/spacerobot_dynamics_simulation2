import numpy as np


def skew_sym(x):
    x = x.flatten()
    x_skew = np.array([[0, -x[2], x[1]],
                       [x[2], 0, -x[0]],
                       [-x[1], x[0], 0]])
    return x_skew

