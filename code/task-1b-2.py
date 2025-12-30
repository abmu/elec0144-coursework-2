
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, SGDMomentum, Adam
from utils import generate_polynomial_data, plot_losses, plot_prediction


configs = []

LRS = [0.1, 0.01, 0.001, 0.0001]

for lr in LRS:
    configs.append(
        (
            [
                (1, None),
                (3, 'tanh'),
                (1, 'linear'),
            ],
            SGD(lr=lr)
        )
    )

iterations = 10000
xs_train, ys_train = generate_polynomial_data(start=-1, stop=1, step=0.05)
xs_test, _ = generate_polynomial_data(start=-0.97, stop=0.93, step=0.1)

losses = []

for lr, (layers, optimiser) in zip(LRS, configs):
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=optimiser
    )

    res = mlp.train(iterations=iterations, train_data=(xs_train, ys_train))
    losses.append((f'lr={lr}', res.train_losses))

    ys_pred = mlp.predict(xs_test)
    plot_prediction(pred=(xs_test, ys_pred), actual=(xs_train, ys_train))

plot_losses(losses)
