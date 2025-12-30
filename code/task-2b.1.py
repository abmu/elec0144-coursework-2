
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD
from utils import parse_classification_data, data_split, plot_loss, plot_acc, plot_prediction, plot_data


# Define network layers: 4 inputs -> 5 hidden (tanh) -> 3 hidden (tanh) -> 3 outputs (linear)
layers = [
    (4, None),
    (5, 'tanh'),
    (3, 'tanh'),
    (3, 'linear'),
]

# Use Stochastic Gradient Descent as per task requirements
optimiser = SGD(lr=0.01)

# Initialize the MLP
mlp = MultilayerPerceptron(
    layers=layers,
    optimiser=optimiser,
    task='classification'
)

# Load and preprocess data
filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)

# Split data: 70% training, 30% validation
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

# Train the network
res = mlp.train(
    iterations=10000,
    train_data=(xs_train, ys_train),
    val_data=(xs_val, ys_val),
    # val_patience=float('inf') 
)

# Visualize results
plot_loss(res.train_losses, res.val_losses)
plot_acc(res.train_accs, res.val_accs)

# Evaluate on validation set (serving as the test set here)
xs_test, ys_test = xs_val, ys_val
ys_pred = mlp.predict(xs_test)

# Convert predictions to class labels
ys_test_cls = mlp.to_classification(ys_test)
ys_pred_cls = mlp.to_classification(ys_pred)

# Plot predictions vs actual
xaxis = range(1, len(xs_test)+1)
plot_prediction(pred=(xaxis, ys_pred_cls), actual=(xaxis, ys_test_cls))

# Plot difference (errors)
difference = ((ys_pred_cls - ys_test_cls) != 0).astype(int)
plot_data(xaxis, difference)
