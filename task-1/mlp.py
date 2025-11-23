import numpy as np
from data import generate_polynomial_data

SEED = 144
rng = np.random.default_rng(SEED)

ITERATIONS = 1

# Get training data
xs, ys = generate_polynomial_data(start=-1, stop=1, step=0.05)
num_points = len(xs)

# Define network layers
layers = [1, 3, 2]

# Initialise weights and biases with random values
# IGNORE ZERO INDEX
weights = [0] * len(layers)
biases = [0] * len(layers)
for i in range(1, len(layers)):
    prev_layer = layers[i-1]
    next_layer = layers[i]
    w = rng.standard_normal((next_layer, prev_layer)) * 0.5
    b = rng.standard_normal((next_layer, 1)) * 0.1
    weights[i] = w
    biases[i] = b


def forward(input):
    input_size = layers[0]
    x = input.reshape(1, input_size)
    


# Train neural network
for epoch in range(1, ITERATIONS+1):
    cost = 0
    for i in range(num_points):
        pass