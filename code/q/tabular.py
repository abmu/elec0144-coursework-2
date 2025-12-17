import math
import random
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

    
    def epsilon(self, episode: int) -> float:
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
    

    def choose_action(self, state: tuple[int, int], episode: int) -> str:
        """
        Chooses an action using an epsilon-greedy policy

        Args:
            state: Current state in the environment
            episode: Current episode

        Returns:
            An action to take
        """
        actions = self.env.ACTIONS.keys()
        if random.random() < self.epsilon(episode):
            # exploration - choose random action
            return random.choice(actions)
        else:
            # exploitation - choose action with highest Q-value
            qs = (self.get_q(state, a) for a in actions)
            max_idx, max_q = max(enumerate(qs), key=lambda x: x[1])
            return actions[max_idx]
    
    
    def update_q(self, state: tuple[int, int], action: str, reward: float, next_state: tuple[int, int]) -> None:
        """
        Update the Q table using the reward given for action chosen in the previous state

        Args:
            state: Previous state in the environment
            action: Pending action to perform
            reward: Reward given for action chosen in the previous state
            next_state: New state after the action was performed
        """
        if self.env.is_terminal():
            best_next = 0.0
        else:
            actions = self.env.ACTIONS.keys()
            best_next = max(self.get_q(next_state, a) for a in actions)
        
        old_q = self.get_q(state, action)

        sample = reward + self.gamma * best_next
        bellman_err = sample - old_q
        self.q_table[(state, action)] = old_q + self.alpha * bellman_err