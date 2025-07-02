import numpy as np #type:ignore
import random

# Configuration and Hyperparameters 
# Main settings for the simulation.

# The learning rate (alpha) determines how much Q-value updates after each step
# A higher value gives more weight to new information
LEARNING_RATE = 0.1

# The discount factor (gamma) -> the importance of future rewards
# Higher value future more important
DISCOUNT_FACTOR = 0.9

# The exploration rate (epsilon) is for the epsilon-greedy strategy (https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning)
# Probability that  agent will choose a random action instead of the best one it knows. Helps the agent discover new, potentially better paths.
EXPLORATION_RATE = 0.1

# The total number of episodes we want to train the agent for.
# An episode is one full run from the start state to a goal or penalty state
EPISODES = 1000

GRID_SIZE = 4

# Grid World Setup 
# Define the rewards for each cell in the 4x4 grid
# 0:  Empty cell
# 1:  Goal state
# -1: A penalty state
REWARD_GRID = np.array([
    [0, 0, 0, 1],
    [0, -1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Starting position for the agent in each episode (row, column)
START_STATE = (3, 0)
 
# Four possible actions
# 0 = Up, 1 = Right, 2 = Down, 3 = Left
NUM_ACTIONS = 4

# --- Q-Table Initialisation ---
# The Q-table stores what the agent has learned.

# Creates a table to store the Q-values for every state-action pair.
# The dimensions are (grid_rows, grid_cols, num_actions).
# Initialise all Q-values to zero because the agent knows nothing at the start
q_table = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))


# Helper Functions

def choose_action(state):
    """
    Decides which action to take from a given state using the epsilon-greedy strategy.
    
    Args:
        state: The current position of the agent (row, col).
        
    Returns:
        An integer representing the action to take (0-3).
    """
    # Generate a random number between 0 and 1
    if random.uniform(0, 1) < EXPLORATION_RATE:
        # Choose a completely random action. This helps the agent find new paths.
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        # Choose the best action based on the current Q-values for this state.
        # returns the index (our action) of the highest value.
        return np.argmax(q_table[state])

def get_next_state(state, action):
    """
    Calculates the agent's next position based on its current state and chosen action.
    It also ensures the agent cannot move outside the grid.
    
    Args:
        state: The current position (row, col)
        action: The chosen action (0-3)
        
    Returns:
        The new position (row, col)
    """
    row, col = state
    
    # Update the row or column based on the action.
    if action == 0:  # Up
        row = max(0, row - 1)
    elif action == 1:  # Right
        col = min(GRID_SIZE - 1, col + 1)
    elif action == 2:  # Down
        row = min(GRID_SIZE - 1, row + 1)
    elif action == 3:  # Left
        col = max(0, col - 1)
        
    return (row, col)

def update_q_value(state, action, reward, next_state):
    """
    Updates the Q-value in the Q-table using the Q-learning formula.
    Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
    
    Args:
        state: The state before the action was taken.
        action: The action that was taken.
        reward: The reward received from taking that action.
        next_state: The new state after the action.
    """
    # Get the current Q-value for the state-action pair (Q(s,a))
    current_q_value = q_table[state][action]
    
    # Find the maximum Q-value for the next stat (max(Q(s',a'))) 
    # The best possible reward the agent can get from the next state
    max_future_q = np.max(q_table[next_state])
    
    # If the next state is a goal or penalty state, there is no future reward
    if REWARD_GRID[next_state] != 0:
        max_future_q = 0.0

    # The "target" is the value Q-value should move towards.
    # reward + discount_factor * max_future_q
    target_q_value = reward + DISCOUNT_FACTOR * max_future_q
    
    # Calculate the new Q-value using the learning rate to scale the update.
    new_q_value = current_q_value + LEARNING_RATE * (target_q_value - current_q_value)
    
    # Update the table with the new Q-value.
    q_table[state][action] = new_q_value

# Main Training Loop 

# A list to keep track of the total reward from each episode.
episode_rewards = []

# Loop for the total number of episodes
for episode in range(EPISODES):
    # Reset the agent's position to the start for the new episode
    state = START_STATE
    done = False
    total_reward = 0
    steps = 0 # To prevent from continous run

    # Loop until the agent reaches a goal or penalty
    while not done and steps < 100:
        # 1. Choose an action
        action = choose_action(state)
        
        # 2. Perform the action and get the next state
        next_state = get_next_state(state, action)
        
        # 3. Get the reward for moving to the new state
        reward = REWARD_GRID[next_state]
        
        # 4. Check if in final state
        done = REWARD_GRID[next_state] != 0
        
        # 5. Update the Q-table with what the agent has learned
        update_q_value(state, action, reward, next_state)
        
        # Update the state for the next loop iteration
        state = next_state
        
        # Add the reward to episode's total
        total_reward += reward
        steps += 1
    
    # Add the final reward for episode to list of rewards
    episode_rewards.append(total_reward)

    # Print  update every 100 episodes
    if (episode + 1) % 100 == 0:
        # Calculate the average reward over the last 100 episode
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1}/{EPISODES} | Average Reward (last 100): {avg_reward:.2f}")

print("\nTraining finished")

# Display Results 

# Display the learned policy on the grid
print("\nLearned Policy (Best move from each cell):")

# Vis arrows
action_symbols = ['↑', '→', '↓', '←']

for r in range(GRID_SIZE):
    row_str = ""
    for c in range(GRID_SIZE):
        # Check for special terminal states first
        if (r,c) == (0, 3):
            row_str += " G  " # Goal
        elif (r,c) == (1, 1):
            row_str += " X  " # Penalty
        else:
            # Find the best action for the current state from the Q-table
            best_action = np.argmax(q_table[(r, c)])
            # Add the arrow symbol for that action to our output string
            row_str += f" {action_symbols[best_action]}  "
    print(row_str)

# Print a summary of the final performance
final_avg_reward = np.mean(episode_rewards[-100:])
success_rate = sum(r > 0 for r in episode_rewards) / EPISODES * 100
print(f"\nFinal 100-episode average reward: {final_avg_reward:.2f}")
print(f"Success rate (reaching goal): {success_rate:.1f}%")