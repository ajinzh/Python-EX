import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the environment grid and rewards for r = 3 and r = -3
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

# Define the transition probabilities
transition_probabilities = {
    'Up': [(0, 1), (-1, 0), (1, 0)],
    'Right': [(1, 0), (0, 1), (0, -1)],
    'Down': [(0, -1), (-1, 0), (1, 0)],
    'Left': [(-1, 0), (0, 1), (0, -1)]
}
probabilities = [0.8, 0.1, 0.1]


# Initialize utilities
def initialize_utilities(grid):
    utilities = np.array(grid, dtype=float)
    return utilities


# Define discount factor
gamma = 0.5


# Define function to get next states and utilities
def get_next_state_utility(state, action, utilities, grid):
    next_utility = 0
    for move, prob in zip(transition_probabilities[action], probabilities):
        next_state = (state[0] + move[0], state[1] + move[1])
        if not (0 <= next_state[0] < len(grid[0]) and 0 <= next_state[1] < len(grid)) or grid[next_state[1]][next_state[0]] is None:
            next_state = state
        next_utility += prob * utilities[next_state[1]][next_state[0]]
    return next_utility


# Value iteration algorithm
def value_iteration(grid, rewards, gamma, iterations=30):
    utilities = initialize_utilities(grid)
    for iteration in range(iterations):
        new_utilities = np.copy(utilities)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if not (i == 0 and j == 2):  # skip terminal state
                    state = (j, i)
                    action_utilities = []
                    for action in transition_probabilities.keys():
                        action_utility = get_next_state_utility(state, action, utilities, grid)
                        action_utilities.append(action_utility)
                    new_utilities[i][j] = rewards[i][j] + gamma * max(action_utilities)
        utilities = new_utilities
        print(f"Iteration {iteration + 1}")
        print(pd.DataFrame(utilities))
        print()
    return utilities


# Perform value iteration for r = 3
print("Value Iteration for r = 3")
utilities_r3 = value_iteration(rewards_r3, rewards_r3, gamma)

# Perform value iteration for r = -3
print("Value Iteration for r = -3")
utilities_rm3 = value_iteration(rewards_rm3, rewards_rm3, gamma)
