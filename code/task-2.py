from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import parse_classification_data, train_val_split, plot_loss, plot_prediction, plot_data


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

xs_train, ys_train, xs_val, ys_val = train_val_split(xs, ys, ratio=0.7)

train_losses, val_losses = mlp.train(
    iterations=10000,
    train_data=(xs_train, ys_train),
    val_data=(xs_val, ys_val),
    val_patience=float('inf')  # max patience for testing purposes
)
plot_loss(train_losses, val_losses)

xtest, ytest = xs_val, ys_val
ypreds = mlp.predict(xtest)

# Convert from 3 outputs to 1 maximum value output -- predicted class
ytest = ytest.argmax(axis=1)
ypreds = ypreds.argmax(axis=1)

xaxis = range(1, len(xtest)+1)
plot_prediction(pred=(xaxis, ypreds), actual=(xaxis, ytest))

difference = ((ypreds - ytest) != 0).astype(int)
plot_data(xaxis, difference)