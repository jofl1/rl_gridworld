import numpy as np #type:ignore
import random
from enum import Enum

#  group all  hyperparameters
class Config:
    # how much the agent learns from new information
    learning_rate = 0.02
    # importance of future rewards (gamma)
    discount_factor = 0.9
    # probability of the agent taking a random action to encourage discovering new paths
    exploration_rate = 0.1
    # total number of training sessions
    episodes = 10000

# Using an Enum
class Action(Enum):
    up = 0
    right = 1
    down = 2
    left = 3

# class defining the environment
class World:
    def __init__(self):
        self.size = 4
        self.start_state = (3, 0)
        # The reward grid defines the goal (+1) and penalties (-1)
        self.reward_grid = np.array([
            [0, 0, 0, 1],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def get_reward(self, state):
        """Returns the reward for entering a given state."""
        return self.reward_grid[state]

    def is_terminal_state(self, state):
        """Checks if an episode should end"""
        return self.reward_grid[state] != 0

    def get_next_state(self, state, action):
        """Calculates the agent's new position"""
        row, col = state
        if action == Action.up.value: row = max(0, row - 1)
        elif action == Action.right.value: col = min(self.size - 1, col + 1)
        elif action == Action.down.value: row = min(self.size - 1, row + 1)
        elif action == Action.left.value: col = max(0, col - 1)
        return (row, col)

# class representing the agent
class Agent:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        # Q-table stores the learned values for every state-action pair
        # initialised to zeros at start
        self.q_table = np.zeros((world.size, world.size, len(Action)))

    def choose_action(self, state):
        """Decides an action using the epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.config.exploration_rate:
            return random.choice(list(Action)).value
        # Otherwise, exploit current knowledge by choosing the best-known action
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # Estimate future rewards by finding the max Q-value for the next state
        # If the next state is disallowed, there is no future reward
        max_future_q = 0.0 if self.world.is_terminal_state(next_state) else np.max(self.q_table[next_state])
        # The target value is what the Q-value should be according to this new information.
        target_q = reward + self.config.discount_factor * max_future_q
        # Update the current Q-value, moving it slightly towards the target
        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q

# class for training process
class Trainer:
    def __init__(self, agent, world, config):
        self.agent = agent
        self.world = world
        self.config = config
        self.episode_rewards = []

    def train(self):
        """Runs the main training loop for a set number of episodes"""
        for episode in range(self.config.episodes):
            state = self.world.start_state  # Reset agent to the start for each new episode
            # --- FIX: Initialise total_reward for the current episode ---
            total_reward = 0
            done = False
            steps = 0  # step limit

            while not done and steps < 100:
                # 1. Agent chooses an action based on its policy
                action = self.agent.choose_action(state)
                # 2. The world determines the outcome of that action
                next_state = self.world.get_next_state(state, action)
                reward = self.world.get_reward(next_state)
                done = self.world.is_terminal_state(next_state)
                # 3. The agent learns from this experience by updating its Q-table
                self.agent.update_q_value(state, action, reward, next_state)
                # 4. Move to the next state for the next step
                state = next_state
                # --- FIX: Add the reward from this step to the episode's total ---
                total_reward += reward
                steps += 1

            # --- FIX: After the episode, append its total reward to the list ---
            self.episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{self.config.episodes} | Avg Reward (last 100): {avg_reward:.2f}")

        print("\nTraining finished.")

    def display_results(self):
        """Visualises the learned policy and prints final performance metrics."""
        print("\nLearned Policy (Best move from each cell):")
        action_symbols = ['↑', '→', '↓', '←']
        for r in range(self.world.size):
            row_str = ""
            for c in range(self.world.size):
                if self.world.get_reward((r, c)) == 1: row_str += " G  "
                elif self.world.get_reward((r, c)) == -1: row_str += " X  "
                else: row_str += f" {action_symbols[np.argmax(self.agent.q_table[(r, c)])]}  "
            print(row_str)

        final_avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = sum(r > 0 for r in self.episode_rewards) / self.config.episodes * 100
        print(f"\nFinal 100-episode average reward: {final_avg_reward:.2f}")
        print(f"Success rate (reaching goal): {success_rate:.1f}%")

# Main execution block
# 1. Create the configuration object.
config = Config()

# 2. Initialise the core components, World and Agent.
world = World()
agent = Agent(world, config)

# 3. Create the Trainer, which manages the interaction between the Agent and World.
trainer = Trainer(agent, world, config)
trainer.train()

# 4. Once training is finished, display the results.
trainer.display_results()
