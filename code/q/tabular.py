import math
import random
from .environment import Environment


class QLearning:
    def __init__(self, env: Environment, alpha: float = 0.1, gamma: float = 0.9, epsilon_decay: float = 0.003) -> None:
        self.q_table = {}  # { (state, action): q-value }
        self.env = env
        self.t = 0

        # hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01

    
    def __str__(self) -> str:
        output = ['===== Q-TABLE =====', '\n']
        actions = self.env.ACTIONS.keys()
        row_format = '{:>8}' * (len(actions) + 2)
        output.append(row_format.format('', *actions, '*BEST*') + '\n')
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                state = (i, j)
                best = self.choose_best_action(state) if not (self.env.is_terminal(state) or self.env.is_obstacle(state)) else ''
                output.append(row_format.format(str(state), *(f'{self.get_q(state, a):.3f}' for a in actions), best) + '\n')
        return ''.join(output)

    
    @property
    def epsilon(self) -> float:
        """
        Get the exponential decay epsilon value

        Returns:
            The epsilon value
        """
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * self.t)
    

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
    

    def choose_train_action(self, state: tuple[int, int]) -> tuple[str, bool]:
        """
        Chooses an action using an epsilon-greedy policy

        Args:
            state: Current state in the environment

        Returns:
            An action to take and whether it is exploitation
        """
        if random.random() < self.epsilon:
            # exploration - choose random action
            actions = self.env.ACTIONS.keys()
            return random.choice(list(actions)), False
        else:
            # exploitation - choose action with highest Q-value
            return self.choose_best_action(state), True
        

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

    
    def run(self, train: bool = False, verbose: bool = False) -> tuple[float, str]:
        """
        Run a tabular Q-learning training episode -- or evaluate the best performance

        Args:
            train: Whether it is training mode or "try-hard" mode
            verbose: Whether to print to stdout or not

        Returns:
            A tuple containing the total reward for that episode and the actions taken
        """
        self.env.reset()
        t_start = self.t

        total_reward = 0.0
        actions = []

        start = self.env.get_pos()
        next = start
        total_reward += self.env.cell_value(next)
        if verbose:
            print(f'Iteration: {self.t - t_start} | Pos: {next} | Epsilon: {self.epsilon:.3f}\n{self}')
        while not self.env.is_terminal(next):
            # Perform action
            state = next
            if train:
                action, is_best = self.choose_train_action(state)
            else:
                action = self.choose_best_action(state)
                is_best = True
            self.env.move(action)
            actions.append(action)

            # Calculate reward
            next = self.env.get_pos()
            reward = self.env.MOVE_PENALTY + self.env.cell_value(next)
            total_reward += reward

            # Update Q-table
            if train:
                self.update_q(state, action, next, reward)

            # Update internal time step
            self.t += 1
            if verbose:
                act_str = '*BEST*' if is_best else '*RANDOM*'
                print(f'Iteration: {self.t - t_start} | Pos: {next} | Epsilon: {self.epsilon:.3f} | Prev. Action: {action} {act_str}\n{self}')

        end = next
        actions_taken = f'START {start} -|   {" -> ".join(actions)}   |- TERMINAL {end}'
        return total_reward, actions_taken
    

    def train(self, episodes: int, seed: int = 42, verbose: bool = False) -> list[float]:
        """
        Train the Q-learning algorithm

        Args:
            seed: Random seed for reproducibility
            episodes: Number of episodes
            verbose: Whether to print to stdout or not

        Returns:
            List of episode rewards
        """
        random.seed(seed)
        self.set_table(table={})
        self.t = 0

        episode_rewards = []

        for episode in range(episodes):
            if verbose:
                print(f'Episode: {episode+1}/{episodes}')
            total_reward, _ = self.run(train=True, verbose=verbose)
            episode_rewards.append(total_reward)

        return episode_rewards