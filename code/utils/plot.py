import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

SAVE = True
OUT_DIR = 'out/'
Path(OUT_DIR).mkdir(exist_ok=True)


def _end() -> None:
    if SAVE:
        fname = OUT_DIR + datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")[:-3] + '.png'
        print(f'Saving to "{fname}"...')
        plt.savefig(fname, dpi=256)
        plt.close()
    else:
        plt.show()


def plot_data(xs: np.ndarray, ys: np.ndarray) -> None:
    plt.figure()
    plt.plot(xs, ys, 'k+')
    _end()


def plot_loss(train_ys: np.ndarray, val_ys: np.ndarray = []) -> None:
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(train_ys)+1), train_ys, label="Training")  # start from 1
    if len(val_ys):
        plt.plot(range(1, len(val_ys)+1), val_ys, label="Validation")
    plt.legend()
    _end()


def plot_acc(train_ys: np.ndarray, val_ys: np.ndarray = []) -> None:
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(1, len(train_ys)+1), train_ys, label="Training")  # start from 1
    if len(val_ys):
        plt.plot(range(1, len(val_ys)+1), val_ys, label="Validation")
    plt.legend()
    _end()


def plot_prediction(pred: tuple[np.ndarray, np.ndarray], actual: tuple[np.ndarray, np.ndarray]) -> None:
    plt.figure()
    plt.plot(*actual, 'k+', label="Actual")
    plt.plot(*pred, 'r-', label="Predicted")
    plt.legend()
    _end()


def plot_losses(train_yss: list[tuple[str, np.ndarray]], log: bool = False) -> None:
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for label, train_ys in train_yss:
        plt.plot(range(1, len(train_ys)+1), train_ys, label=label)  # start from 1
    if log:
        plt.yscale('log')
    plt.legend()
    _end()