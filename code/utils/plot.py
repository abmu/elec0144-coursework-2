import numpy as np
import matplotlib.pyplot as plt


def plot_data(xs: np.ndarray, ys: np.ndarray) -> None:
    plt.figure()
    plt.plot(xs, ys, 'k+')
    plt.show()


def plot_loss(train_ys: np.ndarray, val_ys: np.ndarray = []) -> None:
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(train_ys)+1), train_ys, label="Training")  # start from 1
    if len(val_ys):
        plt.plot(range(1, len(val_ys)+1), val_ys, label="Validation")
    plt.legend()
    plt.show()


def plot_prediction(pred: tuple[np.ndarray, np.ndarray], actual: tuple[np.ndarray, np.ndarray]) -> None:
    plt.figure()
    plt.plot(*actual, 'k+', label="Actual")
    plt.plot(*pred, 'r-', label="Predicted")
    plt.legend()
    plt.show()