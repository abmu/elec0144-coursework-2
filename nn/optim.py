import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    """Abstract base class for optimisers"""

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state
        """
        pass


    @abstractmethod
    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        """
        Update gradients based on internal state
        """
        pass


# Maybe add SGD + momentum optimiser algorithm
class Momentum(Optimiser):
    pass


class SGD(Optimiser):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    
    def reset(self) -> None:
        pass

    
    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        for i in range(1, len(weights)):
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]


class Adam(Optimiser):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr  # learning rate

        # Internal state
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        self.t = 0  # timestep
        
        # Shapes used for internal state initialisation
        self.weight_shapes = []
        self.bias_shapes = []
        self.initialised = False


    def _initialise(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """
        Store the shapes of the weights and biases
        """
        self.weight_shapes = [w.shape if w is not None else None for w in weights]
        self.bias_shapes = [b.shape if b is not None else None for b in biases]
        self.initialised = True
        self.reset()


    def reset(self) -> None:
        self.m_w = [np.zeros(shape) if shape is not None else None for shape in self.weight_shapes]
        self.v_w = [np.zeros(shape) if shape is not None else None for shape in self.weight_shapes]
        self.m_b = [np.zeros(shape) if shape is not None else None for shape in self.bias_shapes]
        self.v_b = [np.zeros(shape) if shape is not None else None for shape in self.bias_shapes]
        self.t = 0


    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        if not self.initialised:
            self._initialise(weights, biases)

        self.t += 1

        for i in range(1, len(weights)):
            # Update momentum
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]

            # Update RMS prop (-- improvement on AdaGrad)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)

            # Compute bias corrected estimates
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Weights and biases update
            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
