import numpy as np
from data import generate_polynomial_data
from plot import plot_data, plot_loss, plot_prediction


SEED = 144
rng = np.random.default_rng(SEED)  # random generator

ITERATIONS = 10000
LEARNING_RATE = 0.001
LOSS_GOAL = 1e-3

# Get training data
xs, ys = generate_polynomial_data(start=-1, stop=1, step=0.05)
x_min, x_max = xs.min(), xs.max()


def normalise_x(x: np.ndarray) -> np.ndarray:
    """
    Normalise the input to between [-1, 1]

    Args:
        input: Input values

    Returns:
        Normalised output values between [-1, 1]
    """
    return 2 * (x - x_min) / (x_max - x_min) - 1


xs_norm = normalise_x(xs)

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
    'sigmoid': lambda x: (s := ACTIVATION['sigmoid'](x)) * (1 - s)
}

# Adam
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8

m_w = [0] * len(layers)
v_w = [0] * len(layers)
m_b = [0] * len(layers)
v_b = [0] * len(layers)

for i in range(1, len(layers)):
    m_w[i] = np.zeros_like(weights[i])
    v_w[i] = np.zeros_like(weights[i])
    m_b[i] = np.zeros_like(biases[i])
    v_b[i] = np.zeros_like(biases[i])


def forward(input: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Performs a forward pass through the neural network.

    Args:
        input: Inputs for the input layer of the neural network

    Returns:
        A cache (in the form of a list of tuples) which contains the pre-activation and activation value of every neuron in each layer of the network
    """
    cache = [0] * len(layers)

    # Input layer
    input_size = layers[0][0]
    x = input.reshape(input_size, 1)
    cache[0] = (np.array([]), x)  # no activation function for first layer

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


def backprop(cache: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Performs a backwards pass through the neural network.

    Args:
        cache: The pre-activation and activation value of every neuron in each layer of the network based on the inputs
        y: Actual truth value outputs for given inputs

    Returns
        The gradients for the weights and biases after the backwards pass
    """
    grad_w = [0] * len(layers)
    grad_b = [0] * len(layers)

    # Output layer
    z, a = cache[-1]
    func = layers[-1][1]
    a_prev = cache[-2][1]

    dL_da = a - y  # derivative of loss function with respect to activation layer
    da_dz = DERIVATIVE[func](z)  # derivative of activation layer with respect to pre-activation values
    delta = dL_da * da_dz
    
    dz_dw = a_prev  # derivative of pre-activation values with respect to weights
    dz_db = np.ones_like(b)  # derivative of pre-activation values with respect to biases
    dz_dprev = weights[-1]  # derivative of last layer pre-activation values with respect to previous activation layer

    # Loss function gradients
    grad_w[-1] = delta @ dz_dw.T
    grad_b[-1] = delta * dz_db
    grad_prev = dz_dprev.T @ delta

    # Hidden layers
    for i in range(len(layers)-2, 0, -1):
        z, a = cache[i]
        func = layers[i][1]
        a_prev = cache[i-1][1]

        dL_da = grad_prev
        da_dz = DERIVATIVE[func](z)
        delta = dL_da * da_dz

        grad_w[i] = delta @ a_prev.T
        grad_b[i] = delta
        grad_prev = weights[i].T @ delta

    return grad_w, grad_b


def stochastic_grad_descent(grad_w: list[np.ndarray], grad_b: list[np.ndarray], lr: float = 0.01) -> None:
    """
    Peforms a stocahstic gradient descent based on the backwards pass result

    Args:
        grad_w: Gradients for the weights after single backwards pass
        grad_b: Gradients for the biases after single backwards pass
        lr: Learning rate of gradient descent
    """
    for i in range(1, len(layers)):
        weights[i] -= lr * grad_w[i]
        biases[i] -= lr * grad_b[i]


def adam_update(grad_w: list[np.ndarray], grad_b: list[np.ndarray], t: int, lr: float = 0.01) -> None:
    """
    Peforms an Adam update step

    Args:
        grad_w: Gradients for the weights after single backwards pass
        grad_b: Gradients for the biases after single backwards pass
        t: Timestep
        lr: Learning rate of gradient descent
    """
    for i in range(1, len(layers)):
        # Update momentum
        m_w[i] = BETA_1 * m_w[i] + (1 - BETA_1) * grad_w[i]
        m_b[i] = BETA_1 * m_b[i] + (1 - BETA_1) * grad_b[i]

        # Update RMS prop (-- improvement on AdaGrad)
        v_w[i] = BETA_2 * v_w[i] + (1 - BETA_2) * (grad_w[i] ** 2)
        v_b[i] = BETA_2 * v_b[i] + (1 - BETA_2) * (grad_b[i] ** 2)

        # Compute bias corrected estimates
        m_w_hat = m_w[i] / (1 - BETA_1 ** t)
        m_b_hat = m_b[i] / (1 - BETA_1 ** t)
        v_w_hat = v_w[i] / (1 - BETA_2 ** t)
        v_b_hat = v_b[i] / (1 - BETA_2 ** t)

        # Weights and biases update
        weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + EPSILON)
        biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + EPSILON)


# Train neural network
losses = []
t = 0
for epoch in range(1, ITERATIONS+1):
    error_sse = 0  # sum of squared errors

    for i, (x, y) in enumerate(zip(xs_norm, ys)):
        t += 1

        cache = forward(x)
        _, y_pred = cache[-1]
        error_sse += loss(y_pred, y)

        grad_w, grad_b = backprop(cache, y)
        # stochastic_grad_descent(grad_w, grad_b, LEARNING_RATE)
        adam_update(grad_w, grad_b, t, LEARNING_RATE)
        
    losses.append(error_sse)

    if error_sse < LOSS_GOAL:
        # Stop training early
        print(f'Epoch: {epoch}, Loss = {error_sse:.4f} < Early stoping threshold = {LOSS_GOAL:.4f}')
        break

    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss = {error_sse:.4f}')

plot_loss(losses)

# Run on test data
xtest, _ = generate_polynomial_data(start=-0.97, stop=0.93, step=0.1)
xtest_norm = normalise_x(xtest)
ypreds = np.array([forward(x)[-1][1].item() for x in xtest_norm])

plot_prediction(pred=(xtest, ypreds), actual=(xs, ys))
