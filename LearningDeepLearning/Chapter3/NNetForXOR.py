import numpy as np

np.random.seed(42)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]  # to randomize the order of selection

# Training examples
X_train = [np.array([-1., -1., -1.]),
           np.array([1., -1., 1.]),
           np.array([1., 1., -1.]),
           np.array([1., 1., 1.])]
y_train = [0., 1., 1., 0.]


def neuron_w(input_count):
    weights = np.zeros(input_count + 1)
    for i in range(1, (input_count + 1)):
        weights[i] = np.random.uniform(-1., 1.)

    return weights


n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]


def show_learning():
    print("Current weights")

    for i, w in enumerate(n_w):
        print("Neuron ", i, ": w0 = ", "%5.2f" % w[0],
              ", w1 = ", "%5.2f" % w[1], ", w2 =",
              "%5.2f" % w[2])
        print("------------------------")


def forward_pass(X):
    global n_y

    n_y[0] = np.tanh(np.dot(n_w[0], X))  # Neuron 0
    n_y[1] = np.tanh(np.dot(n_w[1], X))  # Neuron 1
    n2_inputs = np.array([1., n_y[0], n_y[1]])  # 1.0 is bias
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1. / (1. + np.exp(-z2))


def backward_pass(y_truth):
    global n_error

    error_prime = -(y_truth - n_y[2])  # Derivative of output loss function
    derivative = n_y[2] * (1. - n_y[2])  # Derivative of logistic activation at the output
    n_error[2] = error_prime * derivative

    derivative = 1. - n_y[0] ** 2  # tanh derivative for neuron N1
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    derivative = 1. - n_y[1] ** 2  # tanh derivative for neuron N2
    n_error[1] = 1. - n_w[2][2] * n_error[2] * derivative


def adjust_weights(X):
    global n_w

    n_w[0] -= (X * LEARNING_RATE * n_error[0])
    n_w[1] -= (X * LEARNING_RATE * n_error[1])
    n2_inputs = np.array([1., n_y[0], n_y[1]])  # 1.0 is bias
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])


# Network training loop
all_correct = False
while not all_correct:  # Train until convergence
    all_correct = True
    np.random.shuffle(index_list)
    for i in index_list:
        forward_pass(X_train[i])
        backward_pass(y_train[i])
        adjust_weights(X_train[i])
        show_learning()  # Display updated weights

    for i in range(len(X_train)):  # Checking convergence
        forward_pass(X_train[i])
        print("x1=", "%4.1f" % X_train[i][1], ", x2=", "%4.1f" % X_train[i][2], "y=", "%.4f" % n_y[2])

        if (((y_train[i] < 0.5) and (n_y[2] >= 0.5))
                or ((y_train[i] >= 0.5) and (n_y[2] < 0.5))):
            all_correct = False
