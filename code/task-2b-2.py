
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD
from utils import parse_classification_data, data_split, plot_losses, plot_accuracies


LRS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

layers = [
    (4, None),
    (5, 'tanh'),
    (3, 'tanh'),
    (3, 'linear'),
]

iterations = 10000
val_patience = float('inf')

filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

losses = []
accs = []

for lr in LRS:
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=SGD(lr=lr),
        task='classification'
    )

    res = mlp.train(
        iterations=iterations, 
        train_data=(xs_train, ys_train),
        val_data=(xs_val, ys_val),
        val_patience=val_patience
    )
    label = f'lr={lr}'
    losses.append((label, res.val_losses))
    accs.append((label, res.val_accs))

plot_losses(losses, log=True)
plot_accuracies(accs, log=False)
