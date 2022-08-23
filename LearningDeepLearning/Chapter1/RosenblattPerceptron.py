def compute_perceptron_output(w: list, x: list) -> int:
    '''
    Computes the perceptron output based on weighted inputs.

    :param w: List of weight values
    :param x: List of input values
    :return: An integer -1 or 1 based on the weighted sum.
    '''
    z = 0.
    result = 0

    for i in range(len(w)):
        z += w[i] * x[i]

    if z > 0:
        result = 1
    else:
        result = -1

    return result


W = [-0.6, -0.5]
bias = 1.
X = [1.0, 1.0, 1.0]

W.insert(0, bias)

print(f"{compute_perceptron_output(W, X)=}")

######################################

import numpy as np


def compute_output_vector(w, x):
    z = np.dot(w, x)
    return np.sign(z)


print(f"{compute_output_vector(W, X)=}")
