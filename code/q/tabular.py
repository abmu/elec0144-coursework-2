import math
import random
from .environment import Environment


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

    
    def __str__(self) -> str:
        output = ['===== Q-TABLE =====', '\n']
        actions = self.env.ACTIONS.keys()
        row_format = '{:>8}' * (len(actions) + 1)
        output.append(row_format.format('', *actions) + '\n')
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                state = (i, j)
                output.append(row_format.format(str(state), *(f'{self.get_q(state, a):.3f}' for a in actions)) + '\n')
        return ''.join(output)

    
    def epsilon(self, episode: int) -> float:
        """
        Get the exponential decay epsilon value

        Args:
            episode: Current episode

        Returns:
            The epsilon value
        """
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
    

    def set_table(self, table: dict[tuple[tuple[int, int], str], float]) -> None:
        """
        Set the Q-table

        Args:
            table: New table
        """
        self.q_table = table.copy()

    
    def get_table(self) -> dict[tuple[tuple[int, int], str], float]:
        """
        Get the Q-table

        Returns:
            The Q-table
        """
        return self.q_table.copy()


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
    

    def choose_train_action(self, state: tuple[int, int], episode: int) -> str:
        """
        Chooses an action using an epsilon-greedy policy

        Args:
            state: Current state in the environment
            episode: Current episode

        Returns:
            An action to take
        """
        if random.random() < self.epsilon(episode):
            # exploration - choose random action
            actions = self.env.ACTIONS.keys()
            return random.choice(list(actions))
        else:
            # exploitation - choose action with highest Q-value
            return self.choose_best_action(state)
        

    def choose_best_action(self, state: tuple[int, int]) -> str:
        """
        Choose the best perceived action

        Args:
            state: Current state in the environment

        Returns:
            An action to take
        """
        actions = self.env.ACTIONS.keys()
        qs = (self.get_q(state, a) for a in actions)
        max_idx, max_q = max(enumerate(qs), key=lambda x: x[1])
        return list(actions)[max_idx]
    
    
    def update_q(self, state: tuple[int, int], action: str, next_state: tuple[int, int], reward: float) -> None:
        """
        Update the Q table using the reward given for action chosen in the previous state

        Args:
            state: Previous state in the environment
            action: Pending action to perform
            next_state: New state after the action was performed
            reward: Reward given for action chosen in the previous state
        """
        if self.env.is_terminal(next_state):
            best_next = 0.0
        else:
            actions = self.env.ACTIONS.keys()
            best_next = max(self.get_q(next_state, a) for a in actions)
        
        old_q = self.get_q(state, action)

        sample = reward + self.gamma * best_next
        bellman_err = sample - old_q
        self.q_table[(state, action)] = old_q + self.alpha * bellman_err

    
    def run(self, episode: int = 0, train: bool = False) -> tuple[float, str]:
        """
        Run a tabular Q-learning training episode -- or evaluate the best performance

        Args:
            episode: Current episode
            train: Whether it is training mode or "try-hard" mode

        Returns:
            A tuple containing the total reward for that episode and the actions taken
        """
        self.env.reset()

        total_reward = 0.0
        actions = []

        start = self.env.get_pos()
        next = start
        total_reward += self.env.cell_value(next)
        while not self.env.is_terminal(next):
            # Perform action
            state = next
            if train:
                action = self.choose_train_action(state, episode)
            else:
                action = self.choose_best_action(state)
            self.env.move(action)
            actions.append(action)

            # Calculate reward
            next = self.env.get_pos()
            reward = self.env.MOVE_PENALTY + self.env.cell_value(next)
            total_reward += reward

            # Update Q-table
            if train:
                self.update_q(state, action, next, reward)

        end = next
        actions_taken = f'START {start} -|   {" -> ".join(actions)}   |- TERMINAL {end}'
        return total_reward, actions_taken
    

    def train(self, iterations: int, seed: int = 42) -> list[float]:
        """
        Train the Q-learning algorithm

        Args:
            iterations: Number of iterations

        Returns:
            List of episode rewards
        """
        random.seed(seed)
        self.set_table(table={})

        episode_rewards = []

        for episode in range(iterations):
            total_reward, _ = self.run(episode, train=True)
            episode_rewards.append(total_reward)

        return episode_rewards