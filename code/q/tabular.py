import math
from environment import Environment


class QLearning:
    def __init__(self, env: Environment, alpha: float = 0.1, gamma: float = 0.9, epsilon_decay: float = 0.995) -> None:
        self.q_table = {}  # { (state, action): q-value }
        self.env = env

        # hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01

    
    def epsilon(self, episode: float) -> float:
        """
        Get the exponential decay epsilon value

        Args:
            episode: Current episode

        Returns:
            The epsilon value
        """
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
    

    def get_q(self, state: tuple[int, int], action: str) -> float:
        """
        Get the q-value for a state and action

        Args:
            state: Current state in the environment
            action: Pending action to perform

        Returns:
            Q-value of the state and action combination
        """
        return self.q_table.get((state, action), 0.0)