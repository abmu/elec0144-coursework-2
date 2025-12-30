
# NOTE: The 'q' and 'utils' modules imported below was created entirely by our team
# It is NOT an external package made by someone else! The code can be found in the 'q' and 'utils' folders within this directory

from q import Environment, QLearning
from utils import parse_grid_data, plot_rewards


# Create environment from grid file
filename = 'task-4-grid.txt'
grid = parse_grid_data(filename)
environment = Environment(grid)

# Initialise q-learning algorithm
q_learning = QLearning(
    env=environment,
    alpha=0.1,
    gamma=0.9,
    epsilon_decay=0.995
)

# Train the q-learning algorithm
rewards = q_learning.train(iterations=100)

# Display results
plot_rewards(rewards)

print('NOTE: Coordinates are (Y, X), with origin at top-left corner\n')
print(environment)
print(q_learning)

best_reward, actions_taken = q_learning.run(train=False)
print(f'===== BEST ACTIONS =====\n{actions_taken}\n(Reward: {best_reward})')
