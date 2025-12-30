
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, SGDMomentum, Adam
from utils import parse_classification_data, data_split, plot_losses, plot_accuracies


sgd_configs = []
sgdm_configs = []
adam_configs = []

LRS = [0.1, 0.01, 0.001, 0.0001]

for lr in LRS:
    sgd_configs.append(
        (
            [
                (4, None),
                (5, 'tanh'),
                (3, 'tanh'),
                (3, 'linear'),
            ],
            SGD(lr=lr)
        )
    )
    sgdm_configs.append(
        (
            [
                (4, None),
                (5, 'tanh'),
                (3, 'tanh'),
                (3, 'linear'),
            ],
            SGDMomentum(lr=lr, momentum=0.9)
        )
    )
    adam_configs.append(
        (
            [
                (4, None),
                (5, 'tanh'),
                (3, 'tanh'),
                (3, 'linear'),
            ],
            Adam(lr=lr)
        )
    )

iterations = 3000
val_patience = float('inf')

filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

sgd_losses, sgd_accs = [], []
sgdm_losses, sgdm_accs = [], []
adam_losses, adam_accs = [], []

for i, lr in enumerate(LRS):
    sgd_mlp = MultilayerPerceptron(
        layers=sgd_configs[i][0],
        optimiser=sgd_configs[i][1],
        task='classification'
    )
    sgd_res = sgd_mlp.train(iterations=iterations, train_data=(xs_train, ys_train), val_data=(xs_val, ys_val), val_patience=val_patience)

    sgdm_mlp = MultilayerPerceptron(
        layers=sgdm_configs[i][0],
        optimiser=sgdm_configs[i][1],
        task='classification'
    )
    sgdm_res = sgdm_mlp.train(iterations=iterations, train_data=(xs_train, ys_train), val_data=(xs_val, ys_val), val_patience=val_patience)

    adam_mlp = MultilayerPerceptron(
        layers=adam_configs[i][0],
        optimiser=adam_configs[i][1],
        task='classification'
    )
    adam_res = adam_mlp.train(iterations=iterations, train_data=(xs_train, ys_train), val_data=(xs_val, ys_val), val_patience=val_patience)

    # These plots are using the VALIDATION results, not TRAINING!

    plot_losses([
        (f'SGD', sgd_res.val_losses),
        (f'SGDM', sgdm_res.val_losses),
        (f'ADAM', adam_res.val_losses)
    ], log=True)

    plot_accuracies([
        (f'SGD', sgd_res.val_accs),
        (f'SGDM', sgdm_res.val_accs),
        (f'ADAM', adam_res.val_accs)
    ], log=True)

    sgd_losses.append((f'lr={lr}', sgd_res.val_losses))
    sgd_accs.append((f'lr={lr}', sgd_res.val_accs))
    sgdm_losses.append((f'lr={lr}', sgdm_res.val_losses))
    sgdm_accs.append((f'lr={lr}', sgdm_res.val_accs))
    adam_losses.append((f'lr={lr}', adam_res.val_losses))
    adam_accs.append((f'lr={lr}', adam_res.val_accs))

plot_losses(sgd_losses, log=True)
plot_accuracies(sgd_accs, log=True)
plot_losses(sgdm_losses, log=True)
plot_accuracies(sgdm_accs, log=True)
plot_losses(adam_losses, log=True)
plot_accuracies(adam_accs, log=True)
