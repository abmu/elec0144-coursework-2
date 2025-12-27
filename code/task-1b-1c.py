
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, SGDMomentum, Adam
from utils import generate_polynomial_data, plot_losses


sgd_configs = []
sgdm_configs = []
adam_configs = []

LRS = [0.1, 0.01, 0.001, 0.0001]

for lr in LRS:
    sgd_configs.append(
        (
            [
                (1, None),
                (3, 'tanh'),
                (1, 'linear'),
            ],
            SGD(lr=lr)
        )
    )
    sgdm_configs.append(
        (
            [
                (1, None),
                (3, 'tanh'),
                (1, 'linear'),
            ],
            SGDMomentum(lr=lr, momentum=0.9)
        )
    )
    adam_configs.append(
        (
            [
                (1, None),
                (3, 'tanh'),
                (1, 'linear'),
            ],
            Adam(lr=lr)
        )
    )

iterations = 3000
xs_train, ys_train = generate_polynomial_data(start=-1, stop=1, step=0.05)

sgd_losses = []
sgdm_losses = []
adam_losses = []

for i, lr in enumerate(LRS):
    sgd_mlp = MultilayerPerceptron(
        layers=sgd_configs[i][0],
        optimiser=sgd_configs[i][1]
    )
    sgd_train_losses, _ = sgd_mlp.train(iterations=iterations, train_data=(xs_train, ys_train))

    sgdm_mlp = MultilayerPerceptron(
        layers=sgdm_configs[i][0],
        optimiser=sgdm_configs[i][1]
    )
    sgdm_train_losses, _ = sgdm_mlp.train(iterations=iterations, train_data=(xs_train, ys_train))

    adam_mlp = MultilayerPerceptron(
        layers=adam_configs[i][0],
        optimiser=adam_configs[i][1]
    )
    adam_train_losses, _ = adam_mlp.train(iterations=iterations, train_data=(xs_train, ys_train))

    plot_losses([
        (f'SGD', sgd_train_losses),
        (f'SGDM', sgdm_train_losses),
        (f'ADAM', adam_train_losses)
    ], log=True)

    sgd_losses.append((f'lr={lr}', sgd_train_losses))
    sgdm_losses.append((f'lr={lr}', sgdm_train_losses))
    adam_losses.append((f'lr={lr}', adam_train_losses))

plot_losses(sgd_losses, log=True)
plot_losses(sgdm_losses, log=True)
plot_losses(adam_losses, log=True)
