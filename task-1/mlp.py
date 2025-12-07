import numpy as np
from data import generate_polynomial_data

SEED = 144
rng = np.random.default_rng(SEED)  # random generator

ITERATIONS = 1
LEARNING_RATE = 0.01

# Get training data
xs, ys = generate_polynomial_data(start=-1, stop=1, step=0.05)

# Define network layers
layers = [
    # (size, activation)
    (1, None),
    (3, 'tanh'),
    (1, 'linear'),
]

# Initialise weights and biases with random values
# NOTE: 0-index will not be used
weights = [0] * len(layers)
biases = [0] * len(layers)
for i in range(1, len(layers)):
    prev_layer = layers[i-1][0]
    next_layer = layers[i][0]
    # Glorot initialisation
    limit = np.sqrt(1 / prev_layer)
    w = rng.normal(0, limit, size=(next_layer, prev_layer))
    b = np.zeros((next_layer, 1))
    weights[i] = w
    biases[i] = b

ACTIVATION = {
    'linear': lambda x: x,
    'tanh': lambda x: np.tanh(x),
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x))
}

DERIVATIVE = {
    'linear': lambda x: np.ones_like(x),
    'tanh': lambda x: 1 - np.tanh(x) ** 2,
    'relu': lambda x: (x > 0).astype(x.dtype),
    'sigmoid': lambda x: ACTIVATION['sigmoid'](x) * (1 - ACTIVATION['sigmoid'](x))
}


def forward(input: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Performs a forward pass through the neural network.

    Args:
        input: Inputes for the input layer of the neural network

    Returns:
        A cache (in the form of a list of tuples) which contains the pre-activation and activation value of every neuron in each layer of the network
    """
    cache = [0] * len(layers)

    # Input layer
    input_size = layers[0][0]
    x = input.reshape(input_size, 1)
    cache[0] = (x, np.array([]))  # no activation value for first layer

    # Hidden layers
    for i in range(1, len(layers)-1):
        w = weights[i]
        b = biases[i]
        z = w @ x + b  # pre-activation value of layer
        func = layers[i][1]
        a = ACTIVATION[func](z)
        cache[i] = (z, a)
        x = a

    # Output layer
    w = weights[-1]
    b = biases[-1]
    z = w @ x + b
    func = layers[-1][1]
    a = ACTIVATION[func](z)
    cache[-1] = (z, a)

    return cache


def loss(output: np.ndarray, actual: np.ndarray) -> np.float64:
    """
    Calculate the loss for a single sample

    Args:
        output: Output layer of neural network
        actual: Actual truth value

    Returns
        The loss value calculated between the network output and actual value
    """
    return 0.5 * np.sum((output - actual)**2)


def backprop(cache: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray, lr: float = 0.01) -> None:
    grad_w = [0] * len(layers)
    grad_b = [0] * len(layers)

    # Output layer
    z, a = cache[-1]
    func = layers[-1][1]
    a_prev = cache[-2][1] if len(layers) > 2 else cache[0][0]

    dL_da = a - y  # derivative of loss function with respect to activation layer
    da_dz = DERIVATIVE[func](z)  # derivative of activation layer with respect to pre-activation values
    delta = dL_da * da_dz
    
    dz_dw = a_prev  # derivative of pre-activation values with respect to weights
    dz_db = 1  # derivative of pre-activation values with respect to biases
    dz_dprev = weights[-1]  # derivative of last layer pre-activation values with respect to previous activation layer

    # Loss function gradients
    grad_w[-1] = delta @ dz_dw.T
    grad_b[-1] = delta * dz_db
    grad_prev = dz_dprev.T @ delta

    # Hidden layers
    for i in range(len(layers)-2, 0, -1):
        z, a = cache[i]
        func = layers[i][1]
        a_prev = cache[i-1][1] if i-1 > 0 else cache[0][0]

        dL_da = grad_prev
        da_dz = DERIVATIVE[func](z)
        delta = dL_da * da_dz

        grad_w[i] = delta @ a_prev.T
        grad_b[i] = delta
        grad_prev = weights[i].T @ delta

    # Gradient descent update
    for i in range(1, len(layers)):
        weights[i] -= lr * grad_w[i]
        biases[i] -= lr * grad_b[i]


# Train neural network
for epoch in range(1, ITERATIONS+1):
    error_sse = 0  # sum of squared errors
    for i, (x, y) in enumerate(zip(xs, ys)):
        cache = forward(x)
        _, y_pred = cache[-1]
        error_sse += loss(y_pred, y)
        backprop(cache, y, LEARNING_RATE)