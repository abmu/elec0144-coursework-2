

# TODO
# CHANGE THIS CODE!

# MAKE IT SIMILAR TO 'task-1c.py'










# NOTE: The 'nn' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'nn' and 'utils' folders within this directory

import os
import numpy as np
import matplotlib.pyplot as plt

from nn import MultilayerPerceptron
from nn.optim import SGDMomentum, Adam
from utils import parse_classification_data, data_split



LAYERS = [
    (4, None),
    (5, "tanh"),
    (3, "tanh"),
    (3, "linear"),
]

ITERATIONS = 3000
LRS = [0.1, 0.01, 0.001, 0.0001]
MOMENTUM = 0.9
SEED = 42

OUT_DIR = "task2c_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Smoothing window for plots (visualisation only)
SMOOTH_WINDOW = 50


def build_mlp(optimiser):
    return MultilayerPerceptron(layers=LAYERS, optimiser=optimiser)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def moving_average(y, window):
    if window is None or window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def train_once(optimiser, xs_train, ys_train, xs_val, ys_val):
    np.random.seed(SEED)
    mlp = build_mlp(optimiser)

    train_losses, val_losses = mlp.train(
        iterations=ITERATIONS,
        train_data=(xs_train, ys_train),
        val_data=(xs_val, ys_val),
        val_patience=float("inf"),  # disable early stopping
    )

    train_pred = mlp.predict(xs_train).argmax(axis=1)
    val_pred = mlp.predict(xs_val).argmax(axis=1)

    train_true = ys_train.argmax(axis=1)
    val_true = ys_val.argmax(axis=1)

    train_acc = accuracy(train_true, train_pred)
    val_acc = accuracy(val_true, val_pred)

    return train_losses, val_losses, train_acc, val_acc


def plot_compare_lr(lr, mom_losses, adam_losses, out_path):
    plt.figure()

    mom_plot = moving_average(mom_losses, SMOOTH_WINDOW)
    adam_plot = moving_average(adam_losses, SMOOTH_WINDOW)

    plt.plot(mom_plot, label=f"SGD + Momentum (m={MOMENTUM})")
    plt.plot(adam_plot, label="Adam")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Validation loss comparison (lr={lr})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_lrs(results, title, out_path):
    plt.figure()

    for lr, losses in results.items():
        smoothed = moving_average(losses, SMOOTH_WINDOW)
        plt.plot(smoothed, label=f"lr={lr}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    filename = os.path.join(os.path.dirname(__file__), "task-2-iris.txt")
    xs, ys, _ = parse_classification_data(filename)
    xs_train, ys_train, xs_val, ys_val = data_split(xs, ys, ratio=0.7)

    momentum_results = {}
    adam_results = {}
    summary = []

    print("\nTask 2c – Optimiser comparison (classification)\n")

    for lr in LRS:
        print(f"Learning rate = {lr}")

        # SGD + Momentum
        opt_mom = SGDMomentum(lr=lr, momentum=MOMENTUM)
        _, mom_va, mom_tr_acc, mom_va_acc = train_once(
            opt_mom, xs_train, ys_train, xs_val, ys_val
        )
        momentum_results[lr] = mom_va

        # Adam
        opt_adam = Adam(lr=lr)
        _, adam_va, adam_tr_acc, adam_va_acc = train_once(
            opt_adam, xs_train, ys_train, xs_val, ys_val
        )
        adam_results[lr] = adam_va

        summary.append((lr, mom_tr_acc, mom_va_acc, adam_tr_acc, adam_va_acc))

        # Per-LR comparison plot
        plot_compare_lr(
            lr,
            mom_va,
            adam_va,
            os.path.join(OUT_DIR, f"compare_lr_{lr}.png"),
        )

    # All-LR plots
    plot_all_lrs(
        momentum_results,
        "SGD + Momentum — validation loss for different learning rates",
        os.path.join(OUT_DIR, "momentum_all_lrs.png"),
    )

    plot_all_lrs(
        adam_results,
        "Adam — validation loss for different learning rates",
        os.path.join(OUT_DIR, "adam_all_lrs.png"),
    )

    # Accuracy summary
    print("\n=== Accuracy summary ===")
    print("LR\tMomentum (train/val)\tAdam (train/val)")
    for lr, m_tr, m_va, a_tr, a_va in summary:
        print(f"{lr}\t{m_tr:.3f}/{m_va:.3f}\t\t{a_tr:.3f}/{a_va:.3f}")

    print(f"\nPlots saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()

