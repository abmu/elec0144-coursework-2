
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD, Adam
from utils import parse_classification_data, data_split, plot_loss, plot_acc, plot_prediction, plot_data


layers = [
    (4, None),
    (5, 'tanh'),
    (3, 'tanh'),
    (3, 'linear'),
]

optimiser = Adam(lr=0.001)

mlp = MultilayerPerceptron(
    layers=layers,
    optimiser=optimiser,
    task='classification'
)

filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)

xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

res = mlp.train(
    iterations=10000,
    train_data=(xs_train, ys_train),
    val_data=(xs_val, ys_val),
    # val_patience=float('inf')  # max patience for testing purposes
)
plot_loss(res.train_losses, res.val_losses)
plot_acc(res.train_accs, res.val_accs)

xs_test, ys_test = xs_val, ys_val
ys_pred = mlp.predict(xs_test)

# Convert from 3 outputs to 1 maximum value output -- predicted class
ys_test_cls = mlp.to_classification(ys_test)
ys_pred_cls = mlp.to_classification(ys_pred)

xaxis = range(1, len(xs_test)+1)
plot_prediction(pred=(xaxis, ys_pred_cls), actual=(xaxis, ys_test_cls))

difference = ((ys_pred_cls - ys_test_cls) != 0).astype(int)
plot_data(xaxis, difference)