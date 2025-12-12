import numpy as np
import matplotlib.pyplot as plt


def plot_data(xs: np.ndarray, ys: np.ndarray) -> None:
    plt.figure()
    plt.plot(xs, ys, 'k+')
    plt.show()


def plot_loss(ys: np.ndarray) -> None:
    plt.figure()
    plt.plot(range(1, len(ys)+1), ys)  # start from 1
    plt.show()


def plot_prediction(pred: tuple[np.ndarray, np.ndarray], actual: tuple[np.ndarray, np.ndarray]) -> None:
    plt.figure()
    plt.plot(*actual, 'k+')
    plt.plot(*pred, 'r-')
    plt.show()