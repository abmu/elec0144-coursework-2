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
    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Update weights and biases based on internal state and gradients
        """
        pass


class SGD(Optimiser):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    
    def reset(self) -> None:
        pass

    
    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        for i in range(1, len(weights)):
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]

        return weights, biases


class SGDMomentum(Optimiser):
    def __init__(self, lr: float = 0.001, momentum: float = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum

        # Velocity terms
        self.v_w = [None]
        self.v_b = [None]

        self.initialised = False

    
    def _initialise(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """
        Initialise velocity terms
        """
        n = len(weights)

        self.v_w = [None] + [np.zeros_like(weights[i]) for i in range(1, n)]
        self.v_b = [None] + [np.zeros_like(biases[i]) for i in range(1, n)]

        self.initialised = True


    def reset(self) -> None:
        if not self.initialised:
            return

        self.initialised = False


    def update(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        grad_w: list[np.ndarray],
        grad_b: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:

        if not self.initialised:
            self._initialise(weights, biases)

        for i in range(1, len(weights)):
            # Update velocity
            self.v_w[i] = self.momentum * self.v_w[i] + grad_w[i]
            self.v_b[i] = self.momentum * self.v_b[i] + grad_b[i]

            # Update parameters
            weights[i] -= self.lr * self.v_w[i]
            biases[i] -= self.lr * self.v_b[i]

        return weights, biases


class Adam(Optimiser):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr  # learning rate

        # Internal state
        self.m_w = [None,]
        self.v_w = [None,]
        self.m_b = [None,]
        self.v_b = [None,]
        self.t = 0  # timestep
        
        self.initialised = False


    def _initialise(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """
        Initialise the internal state variables
        """
        n = len(weights)

        self.m_w = [None] + [np.zeros_like(weights[i]) for i in range(1, n)]
        self.v_w = [None] + [np.zeros_like(weights[i]) for i in range(1, n)]
        self.m_b = [None] + [np.zeros_like(biases[i]) for i in range(1, n)]
        self.v_b = [None] + [np.zeros_like(biases[i]) for i in range(1, n)]

        self.initialised = True


    def reset(self) -> None:
        if not self.initialised:
            return
        
        self.t = 0
        self.initialised = False        


    def update(self, weights: list[np.ndarray], biases: list[np.ndarray], grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.initialised:  # lazy initialisation
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

        return weights, biases