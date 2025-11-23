import numpy as np
from data import generate_polynomial_data

SEED = 144
rng = np.random.default_rng(SEED)  # random generator

ITERATIONS = 1

# Get training data
xs, ys = generate_polynomial_data(start=-1, stop=1, step=0.05)

# Define network layers
layers = [1, 3, 1]

# Initialise weights and biases with random values
# NOTE: 0-index will not be used
weights = [0] * len(layers)
biases = [0] * len(layers)
for i in range(1, len(layers)):
    prev_layer = layers[i-1]
    next_layer = layers[i]
    w = rng.normal(size=(next_layer, prev_layer), scale=0.5)
    b = rng.normal(size=(next_layer, 1), scale=0.1)
    weights[i] = w
    biases[i] = b


def activation(x: np.ndarray, func: str = 'linear') -> np.ndarray:
    """
    Applies activation function to input values

    Args:
        x: Input values
        func: Desired activation function

    Returns:
        The result of func(x)
    """
    if func == 'linear':
        return x
    elif func == 'tanh':
        return np.tanh(x)
    # default to linear function
    return x


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
    input_size = layers[0]
    x = input.reshape(input_size, 1)
    cache[0] = (x, np.array([]))

    # Hidden layers
    for i in range(1, len(layers)-1):
        w = weights[i]
        b = biases[i]
        z = w @ x + b  # pre-activation value of layer
        a = activation(z, 'tanh')
        cache[i] = (z, a)
        x = a

    # Output layer
    w = weights[-1]
    b = biases[-1]
    z = w @ x + b
    a = activation(z, 'linear')
    cache[-1] = (z, a)

    return cache


def loss(output: np.ndarray, actual: np.ndarray):
    """
    Calculate the loss for a single sample

    Args:
        output: Output layer of neural network
        actual: Actual truth value

    Returns
        The loss value calculated between the network output and actual value
    """
    return 0.5 * np.sum((output - actual)**2)


# Train neural network
for epoch in range(1, ITERATIONS+1):
    error_sse = 0  # sum of squared errors
    for i, (x, y) in enumerate(zip(xs, ys)):
        cache = forward(x)
        y_pred = cache[-1]
        error_sse += loss(y_pred, y)