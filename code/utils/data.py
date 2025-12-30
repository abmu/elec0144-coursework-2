import numpy as np


def generate_polynomial_data(
        start: float = -1,
        stop: float = 1,
        step: float = 0.05,
        mean: float = 0,
        spread: float = 0.02,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate polynomial regression data with added Gaussian noise.

    Args:
        seed: Random seed for reproducibility
        start: Start value for x range
        stop: Stop value for x range
        step: Step size for x values
        mean: Mean of the Gaussian noise
        spread: Standard deviation of the Gaussian noise

    Returns:
        Tuple of x and y values as numpy arrays
    """
    rng = np.random.default_rng(seed)
    num = round(1 + (stop - start) / step)
    xs = np.linspace(start, stop, num)
    # Generate y values from polynomial -- as shown in coursework specification
    ys = 0.8 * xs**3 + 0.3 * xs**2 + -0.4 * xs + rng.normal(mean, spread, num)
    return xs, ys


def data_split(xs: np.ndarray, ys: np.ndarray, ratio: float = 0.7, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the input dataset into two sections

    Args:
        seed: Random seed for reproducibility
        xs: Input values
        ys: Truth output values
        ratio: Desired split -- e.g. 0.7 means a 70% / 30% split

    Returns:
        The result of the original dataset being split into two
    """
    # Shuffle data
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(xs))
    xs, ys = xs[idx], ys[idx]

    # Split data
    split_idx = int(ratio * len(xs))
    xs_1, xs_2 = xs[:split_idx], xs[split_idx:]
    ys_1, ys_2 = ys[:split_idx], ys[split_idx:]

    return xs_1, ys_1, xs_2, ys_2


def parse_classification_data(filename: str, epsilon: float = 0.1) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """
    Parse the data within a file to get an input and output values dataset

    Args:
        filename: Name of the file
        epsilon: Value used when encoding the labels as numeric values -- smooth one-hot

    Returns:
        The input and truth output values, and a dictionary that can be used to convert the indexes into corresponding labels
    """
    xs = []
    ys_labels = []

    # Read data from file
    with open(filename, 'r') as f:
        for line in f:
            *features, label = line.strip().split(',')
            xs.append(list(map(float, features)))
            ys_labels.append(label)

    xs = np.array(xs)

    # Convert labels to index value
    unique_labels = sorted(set(ys_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # Smooth one-hot (represent label as a vector where one element is "on" and all others are "off")
    ys = np.full((len(ys_labels), num_classes), epsilon / (num_classes - 1))
    for i, label in enumerate(ys_labels):
        ys[i, label_to_idx[label]] = 1.0 - epsilon
    
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    return xs, ys, idx_to_label


def parse_grid_data(filename: str) -> list[list[str]]:
    """
    Parse the data within a file to get the grid-world

    Args:
        filename: Name of the file

    Returns:
        The grid-world encoded as a 2D array
    """
    grid = []

    # Read data from file
    with open(filename, 'r') as f:
        for line in f:
            grid.append(line.strip().split(','))
    
    return grid


def create_confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    """
    Create a confusion matrix from the true values and predicted values

    Args:
        y_true: True values
        y_pred: Predicted values
        num_classes: Number of classes

    Returns:
        A confusion matrix
    """

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm