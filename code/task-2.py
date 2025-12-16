from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import parse_classification_data, data_split, plot_loss, plot_prediction, plot_data

# TODO
# Measure and display accuracy of neural network during training -- Numner of correct predictions / Total predictions


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
xs, ys, idx_to_label = parse_classification_data(filename)

xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

train_losses, val_losses = mlp.train(
    iterations=10000,
    train_data=(xs_train, ys_train),
    val_data=(xs_val, ys_val),
    # val_patience=float('inf')  # max patience for testing purposes
)
plot_loss(train_losses, val_losses)

xs_test, ys_test = xs_val, ys_val
ys_pred = mlp.predict(xs_test)

# Convert from 3 outputs to 1 maximum value output -- predicted class
ys_test = ys_test.argmax(axis=1)
ys_pred = ys_pred.argmax(axis=1)

xaxis = range(1, len(xs_test)+1)
plot_prediction(pred=(xaxis, ys_pred), actual=(xaxis, ys_test))

difference = ((ys_pred - ys_test) != 0).astype(int)
plot_data(xaxis, difference)