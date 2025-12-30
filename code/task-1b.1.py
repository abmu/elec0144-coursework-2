
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, SGDMomentum, Adam
from utils import generate_polynomial_data, plot_loss, plot_prediction, plot_data


# Define network layers and optimiser
layers = [
    # (size, activation)
    (1, None),
    (3, 'tanh'),
    (1, 'linear'),
]

optimiser = SGD(lr=0.01)
# optimiser = Adam(lr=0.001)

# Setup neural network
mlp = MultilayerPerceptron(
    layers=layers,
    optimiser=optimiser
)

# Get training data
xs_train, ys_train = generate_polynomial_data(start=-1, stop=1, step=0.05)
plot_data(xs_train, ys_train)

# Evaluate losses
res = mlp.train(iterations=10000, train_data=(xs_train, ys_train))
plot_loss(res.train_losses)

# Run on test data
xs_test, _ = generate_polynomial_data(start=-0.97, stop=0.93, step=0.1)
ys_pred = mlp.predict(xs_test)
plot_prediction(pred=(xs_test, ys_pred), actual=(xs_train, ys_train))
