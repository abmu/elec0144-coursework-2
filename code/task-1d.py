
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, SGDMomentum, Adam
from utils import generate_polynomial_data, plot_losses, plot_prediction


configs = [
    ('Tanh', [
        (1, None),
        (3, 'tanh'),
        (1, 'linear'),
    ]),
    ('Tanh-wide', [
        (1, None),
        (9, 'tanh'),
        (1, 'linear'),
    ]),
    ('Tanh-deep', [
        (1, None),
        (3, 'tanh'),
        (3, 'tanh'),
        (1, 'linear'),
    ]),
    ('ReLu-wide-deep', [
        (1, None),
        (9, 'relu'),
        (9, 'relu'),
        (1, 'linear'),
    ]),
    ('ReLu-Tanh-wide-deep', [
        (1, None),
        (9, 'relu'),
        (3, 'tanh'),
        (1, 'linear'),
    ]),
    ('ReLu-output', [
        (1, None),
        (3, 'tanh'),
        (1, 'relu'),
    ]),
    ('Sigmoid', [
        (1, None),
        (3, 'sigmoid'),
        (1, 'linear'),
    ]),
]

optimiser = SGD(lr=0.01)

xs_train, ys_train = generate_polynomial_data(start=-1, stop=1, step=0.05)
xs_test, _ = generate_polynomial_data(start=-0.97, stop=0.93, step=0.1)

losses = []

for label, layers in configs:
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=optimiser
    )

    train_losses, _ = mlp.train(iterations=10000, train_data=(xs_train, ys_train))
    losses.append((label, train_losses))

    ys_pred = mlp.predict(xs_test)
    plot_prediction(pred=(xs_test, ys_pred), actual=(xs_train, ys_train))

plot_losses(losses)
