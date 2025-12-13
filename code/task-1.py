from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import generate_polynomial_data, plot_loss, plot_prediction


# Get training data
xs, ys = generate_polynomial_data(start=-1, stop=1, step=0.05)

# Define network layers and optimiser
layers = [
    # (size, activation)
    (1, None),
    (3, 'tanh'),
    (1, 'linear'),
]

optimiser = Adam(lr=0.001)

# Train and evaluate neural network
mlp = MultilayerPerceptron(
    layers=layers,
    optimiser=optimiser
)

losses = mlp.train(xs, ys, iterations=10000)
plot_loss(losses)

# Run on test data
xtest, _ = generate_polynomial_data(start=-0.97, stop=0.93, step=0.1)
ypreds = mlp.predict(xtest)
plot_prediction(pred=(xtest, ypreds), actual=(xs, ys))
