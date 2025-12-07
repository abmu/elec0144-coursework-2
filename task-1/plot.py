import numpy as np
import matplotlib.pyplot as plt


def plot_data(xs: np.ndarray, ys: np.ndarray) -> None:
    plt.figure()
    plt.plot(xs, ys, 'k+')
    plt.show()