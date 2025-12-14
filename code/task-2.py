from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import parse_classification_data, train_val_split


layers = [
    (4, None),
    (5, 'tanh'),
    (3, 'tanh'),
    (3, 'linear'),
]

optimiser = Adam(lr=0.001)

mlp = MultilayerPerceptron(
    layers=layers,
    optimiser=optimiser
)

filename = 'task-2-iris.txt'
xs, ys, label_to_idx = parse_classification_data(filename)

xs_train, ys_train, xs_val, ys_val = train_val_split(xs, ys, ratio=0.7)

train_losses, val_losses = mlp.train(
    iterations=10000,
    train_data=(xs_train, ys_train),
    val_data=(xs_val, ys_val)
)