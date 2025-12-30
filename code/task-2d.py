
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD
from utils import parse_classification_data, data_split, plot_losses, plot_accuracies


configs = [
    (
        'Tanh',
        [
            (4, None),
            (5, 'tanh'),
            (3, 'tanh'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'Tanh-shallow',
        [
            (4, None),
            (3, 'tanh'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'Tanh-wide',
        [
            (4, None),
            (10, 'tanh'),
            (10, 'tanh'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'ReLU',
        [
            (4, None),
            (5, 'relu'),
            (3, 'relu'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'ReLU-deep',
        [
            (4, None),
            (8, 'relu'),
            (8, 'relu'),
            (8, 'relu'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'Sigmoid',
        [
            (4, None),
            (5, 'sigmoid'),
            (3, 'sigmoid'),
            (3, 'linear'),
        ],
        SGD(lr=0.01)
    ),
    (
        'Sigmoid-output',
        [
            (4, None),
            (5, 'tanh'),
            (3, 'tanh'),
            (3, 'sigmoid'),
        ],
        SGD(lr=0.01)
    ),
]

iterations = 3000
val_patience = float('inf')

filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

losses = []
accs = []

for label, layers, optimiser in configs:
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=optimiser,
        task='classification'
    )

    res = mlp.train(
        iterations=iterations, 
        train_data=(xs_train, ys_train), 
        val_data=(xs_val, ys_val),
        val_patience=val_patience
    )
    losses.append((label, res.val_losses))
    accs.append((label, res.val_accs))

plot_losses(losses, log=True)
plot_accuracies(accs, log=False)
