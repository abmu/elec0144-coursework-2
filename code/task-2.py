from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import parse_classification_data


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

# train_losses, val_losses = mlp.train(xs, ys, iterations=10000)