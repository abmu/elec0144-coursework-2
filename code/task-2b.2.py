
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import SGD
from utils import parse_classification_data, data_split, plot_losses, plot_accuracies


# Define Learning Rates to explore
LRS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

# Fixed architecture for Task 2
layers = [
    (4, None),
    (5, 'tanh'),
    (3, 'tanh'),
    (3, 'linear'),
]

# Load and split data
filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

iterations = 5000
losses = []
accs = []

print(f"Comparing SGD with Learning Rates: {LRS}")

for lr in LRS:
    print(f"--- Training with lr={lr} ---")
    
    # Initialize MLP with current learning rate
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=SGD(lr=lr),
        task='classification'
    )

    # Train
    res = mlp.train(
        iterations=iterations, 
        train_data=(xs_train, ys_train),
        val_data=(xs_val, ys_val),
        val_patience=float('inf') # Train fully to observe convergence behavior
    )

    # Collect results for plotting
    label = f'lr={lr}'
    losses.append((label, res.val_losses))
    accs.append((label, res.val_accs))

# Plot comparisons
plot_losses(losses, log=True)
plot_accuracies(accs, log=False)
