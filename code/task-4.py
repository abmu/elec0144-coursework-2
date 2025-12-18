from q import Environment, QLearning
from utils import parse_grid_data, plot_acc


filename = 'task-4-grid.txt'
grid = parse_grid_data(filename)
environment = Environment(grid)

q_learning = QLearning(
    env=environment,
    alpha=0.1,
    gamma=0.9,
    epsilon_decay=0.995
)

rewards = q_learning.train(iterations=100)
plot_acc(rewards)

_, actions_taken = q_learning.run(train=False)
print(actions_taken)