
# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

from nn import MultilayerPerceptron
from nn.optim import Adam
from utils import parse_classification_data, data_split, plot_losses, plot_accuracies


# Define configurations to explore:
# 1. Baseline (from Task 2): Tanh hidden, Linear output
# 2. ReLU Hidden: ReLU hidden, Linear output (Requirement)
# 3. Sigmoid Hidden: Sigmoid hidden, Linear output
# 4. Small: Fewer nodes
# 5. Wide: More nodes
# 6. Deep ReLU: More layers with ReLU

configs = [
    (
        'Baseline (Tanh)',
        [
            (4, None),
            (5, 'tanh'),
            (3, 'tanh'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
    (
        'ReLU Hidden',
        [
            (4, None),
            (5, 'relu'),
            (3, 'relu'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
    (
        'Sigmoid Hidden',
        [
            (4, None),
            (5, 'sigmoid'),
            (3, 'sigmoid'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
    (
        'Sigmoid Output',
        [
            (4, None),
            (5, 'tanh'),
            (3, 'tanh'),
            (3, 'sigmoid'),
        ],
        Adam(lr=0.001)
    ),
    (
        'Small (Tanh)',
        [
            (4, None),
            (3, 'tanh'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
    (
        'Wide (Tanh)',
        [
            (4, None),
            (10, 'tanh'),
            (10, 'tanh'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
    (
        'Deep ReLU',
        [
            (4, None),
            (8, 'relu'),
            (8, 'relu'),
            (8, 'relu'),
            (3, 'linear'),
        ],
        Adam(lr=0.001)
    ),
]

# Load and split data
filename = 'task-2-iris.txt'
xs, ys, idx_to_label = parse_classification_data(filename)
xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

iterations = 5000
val_losses_list = []
val_accs_list = []

print(f"Training {len(configs)} configurations for {iterations} iterations each...")

for label, layers, optimiser in configs:
    print(f"--- Training {label} ---")
    mlp = MultilayerPerceptron(
        layers=layers,
        optimiser=optimiser,
        task='classification'
    )

    res = mlp.train(
        iterations=iterations, 
        train_data=(xs_train, ys_train), 
        val_data=(xs_val, ys_val),
        val_patience=float('inf') # Train fully to see behavior
    )
    
    val_losses_list.append((label, res.val_losses))
    val_accs_list.append((label, res.val_accs))

# Plot comparisons
plot_losses(val_losses_list, log=True)
plot_accuracies(val_accs_list, log=False)
