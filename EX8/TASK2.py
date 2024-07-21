import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definition the enviropment grid and rewards

rewards_r3 = [
    [3, -1, 10],
    [-1, -1, -1],
    [-1, -1, -1],
]
rewards_rm3 = [
    [-3, -1, 10],
    [-1, -1, -1],
    [-1, -1, -1],
]

# define the transition probabilities
transition_probabilities = {
    'Up': [(0, 1), (-1, 0), (1, 0)],
    'Right': [(1, 0), (0, 1), (0, -1)],
    'Left': [(-1, 0), (0, 1), (0, -1)],
    'Down': [(0, -1), (-1, 0), (1, 0)]
}

probabilities = [0.8, 0.1, 0.1]

# define utilities


def initialize_utilities(grid):
    utilities = np.array(grid, dtype=float)
    return utilities


gamma = 0.5


def update_utility(state, action, utilities, grid):
    next_utility = 0
    for move, prob in zip(transition_probabilities[action], probabilities):
        next_state = (state[0]+move[0], state[1]+move[1])
        if not (0 <= next_state[0] < len(grid[0]) and 0 <= next_state[1] < len(grid)) or grid[next_state[1]][next_state[0]] is None:
            next_state = state
        next_utility += prob * utilities[next_state[1]][next_state[0]]
    return next_utility


def value_iteration(grid, rewards, gamma, max_iterations=30):
    utilities = initialize_utilities(grid)
    utility_history = {state: [] for state in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]}

    for _ in range(max_iterations):
        new_utilities = np.copy(utilities)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not (i == 0 and j == 2):  # skip terminal state
                    state = (j, i)
                    action_utilities = []
                    for action in transition_probabilities.keys():
                        action_utility = update_utility(state, action, utilities, grid)
                        action_utilities.append(action_utility)
                    new_utilities[i][j] = rewards[i][j] + gamma * max(action_utilities)
        utilities = new_utilities
        for state in utility_history:
            utility_history[state].append(utilities[state[1]][state[0]])

    return utilities, utility_history


# Perform value iteration for r = 3
print("Value Iteration for r = 3")
utilities_r3, history_r3 = value_iteration(rewards_r3, rewards_r3, gamma)

# Perform value iteration for r = -3
print("Value Iteration for r = -3")
utilities_rm3, history_rm3 = value_iteration(rewards_rm3, rewards_rm3, gamma)


# Plotting the results

def plot_convergence(history, title):
    for state, utilities in history.items():
        plt.plot(utilities, label=f"State {state}")
    plt.xlabel("Number of iterations")
    plt.ylabel("Utility estimates")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_convergence(history_r3, "Converged utilities for r = 3")
plot_convergence(history_rm3, "Converged utilities for r = -3")


