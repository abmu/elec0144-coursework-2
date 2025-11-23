import numpy as np
import matplotlib.pyplot as plt

SEED = 42
START, STOP, STEP = -1, 1, 0.05
MEAN, SPREAD = 0, 0.02 # mean and standard deviation (spread) parameters for normal distribution

rng = np.random.default_rng(SEED)

num = round(1 + (STOP - START) / STEP)
xs = np.linspace(START, STOP, num)

ys = 0.8 * xs**3 + 0.3 * xs**2 - 0.4 * xs + rng.normal(MEAN, SPREAD, num)

plt.figure()
plt.plot(xs, ys, 'k+')
plt.show()