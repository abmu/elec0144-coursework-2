# task1c_better_plots.py
# comparison of SGD + Momentum vs Adam

import os
import numpy as np
import matplotlib.pyplot as plt

from nn import MultilayerPerceptron
from nn.optim import SGDMomentum, Adam
from utils import generate_polynomial_data


# Define network layers and optimiser
LAYERS = [
    # (size, activation)
    (1, None),
    (3, "tanh"),
    (1, "linear"),
]


ITERATIONS = 3000

# learning rates 
LRS = [0.1, 0.01, 0.001, 0.0001]  

MOMENTUM = 0.9
SEED = 42
OUT_DIR = "task1c_plots"

# Smoothing window for nicer curves (set None to disable)
SMOOTH_WINDOW = 50

# If loss goes NaN/Inf, stop early (saves time)
STOP_ON_INVALID = True


def build_mlp(optimiser):
    return MultilayerPerceptron(layers=LAYERS, optimiser=optimiser)


def moving_average(y, window):
    if window is None or window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def train_once(optimiser, xs_train, ys_train):
    """
    Trains a fresh model and returns:
      losses (np.array), valid_run (bool)
    """
    np.random.seed(SEED)
    mlp = build_mlp(optimiser)

    # Train
    train_losses, _ = mlp.train(iterations=ITERATIONS, train_data=(xs_train, ys_train))
    losses = np.array(train_losses, dtype=float)

    if STOP_ON_INVALID and (np.any(~np.isfinite(losses))):
        # Truncate at first invalid value (keeps plots readable)
        first_bad = np.where(~np.isfinite(losses))[0][0]
        losses = losses[:first_bad]
        return losses, False

    return losses, True


def plot_compare_lr(lr, mom_losses, mom_ok, adam_losses, adam_ok, out_path):
    plt.figure()

    # Smooth for readability
    mom_plot = moving_average(mom_losses, SMOOTH_WINDOW)
    adam_plot = moving_average(adam_losses, SMOOTH_WINDOW)

    label_m = f"SGD + Momentum (m={MOMENTUM})" + ("" if mom_ok else " [diverged]")
    label_a = "Adam" + ("" if adam_ok else " [diverged]")

    plt.plot(mom_plot, label=label_m)
    plt.plot(adam_plot, label=label_a)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss comparison at lr={lr} (smoothed window={SMOOTH_WINDOW})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_all_lrs(losses_by_lr, ok_by_lr, title, out_path, use_log_y=True):
    plt.figure()

    for lr in losses_by_lr:
        y = moving_average(losses_by_lr[lr], SMOOTH_WINDOW)
        tag = "" if ok_by_lr[lr] else " (diverged)"
        plt.plot(y, label=f"lr={lr}{tag}")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title + f" (smoothed window={SMOOTH_WINDOW})")
    if use_log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def final_loss(losses):
    if len(losses) == 0:
        return float("nan")
    return float(losses[-1])



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xs_train, ys_train = generate_polynomial_data(start=-1, stop=1, step=0.05)

    momentum_results = {}
    adam_results = {}
    momentum_ok = {}
    adam_ok = {}

    summary = []

    print("\n Task 1c (SGD Momentum vs Adam) \n")

    for lr in LRS:
        # Momentum
        opt_mom = SGDMomentum(lr=lr, momentum=MOMENTUM)
        mom_losses, mom_ok = train_once(opt_mom, xs_train, ys_train)
        momentum_results[lr] = mom_losses
        momentum_ok[lr] = mom_ok

        # Adam
        opt_adam = Adam(lr=lr)
        adam_losses, adam_ok_run = train_once(opt_adam, xs_train, ys_train)
        adam_results[lr] = adam_losses
        adam_ok[lr] = adam_ok_run

        # Save comparison plot per LR
        compare_path = os.path.join(OUT_DIR, f"compare_lr_{lr}.png")
        plot_compare_lr(lr, mom_losses, mom_ok, adam_losses, adam_ok_run, compare_path)

        mom_final = final_loss(mom_losses)
        adam_final = final_loss(adam_losses)
        summary.append((lr, mom_final, mom_ok, adam_final, adam_ok_run))

        print(f"lr={lr:<7} | Momentum final={mom_final:.6f} {'OK' if mom_ok else 'DIV'}"
              f" | Adam final={adam_final:.6f} {'OK' if adam_ok_run else 'DIV'}")

    # All-LR plots (log scale helps readability)
    plot_all_lrs(
        momentum_results, momentum_ok,
        title=f"SGD + Momentum (m={MOMENTUM}) — loss for different learning rates",
        out_path=os.path.join(OUT_DIR, "momentum_all_lrs.png"),
        use_log_y=True
    )

    plot_all_lrs(
        adam_results, adam_ok,
        title="Adam — loss for different learning rates",
        out_path=os.path.join(OUT_DIR, "adam_all_lrs.png"),
        use_log_y=True
    )

    print("\n=== Summary (lower final loss is better, divergence marked) ===")
    print("lr\t\tMomentum\t\tAdam")
    for lr, mom_final, mom_ok, adam_final, adam_ok_run in summary:
        mom_tag = "" if mom_ok else " (diverged)"
        adam_tag = "" if adam_ok_run else " (diverged)"
        print(f"{lr}\t\t{mom_final:.6f}{mom_tag}\t\t{adam_final:.6f}{adam_tag}")

    print(f"\nPlots saved to: ./{OUT_DIR}/")
    print("Key files:")
    print(" - momentum_all_lrs.png")
    print(" - adam_all_lrs.png")
    print(" - compare_lr_<lr>.png (one per learning rate)\n")


if __name__ == "__main__":
    main()
