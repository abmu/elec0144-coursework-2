import numpy as np
from plot import plot_data


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


if __name__ == "__main__":
    xs, ys = generate_polynomial_data()
    plot_data(xs, ys)