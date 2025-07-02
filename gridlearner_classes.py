import numpy as np #type:ignore
import random
from enum import Enum
import matplotlib.pyplot as plt

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
            state = self.world.start_state
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                action = self.agent.choose_action(state)
                next_state = self.world.get_next_state(state, action)
                reward = self.world.get_reward(next_state)
                done = self.world.is_terminal_state(next_state)
                self.agent.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1

            self.episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{self.config.episodes} | Avg Reward (last 100): {avg_reward:.2f}")

        print("\nTraining finished.")

    def plot_results(self):
        """Creates and displays plots for the learning curve and Q-value directions."""
        # Create a figure with two subplots, arranged side-by-side.
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Q-Learning Performance Analysis', fontsize=16)

        # 1. Learning Curve Plot
        moving_avg_rewards = [np.mean(self.episode_rewards[i:i+100]) for i in range(0, len(self.episode_rewards), 100)]
        episodes_chunked = list(range(100, self.config.episodes + 1, 100))

        axs[0].plot(episodes_chunked, moving_avg_rewards, color='steelblue')
        axs[0].set_title('Learning Curve')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Average Reward (per 100 episodes)')
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # 2. Q-Value Directional Plot
        # shows the agent's preference for each action in every state
        # by displaying the numerical Q-value for each direction.
        axs[1].set_title('Q-Values per State-Action')
        axs[1].set_aspect('equal')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].invert_yaxis() # Match grid orientation

        for r in range(self.world.size):
            for c in range(self.world.size):
                # Draw grid lines for each cell
                axs[1].add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=0.5))

                # Mark Goal and Penalty states with 'G' and 'X'.
                if self.world.get_reward((r, c)) == 1:
                    axs[1].text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20, color='green')
                    continue
                if self.world.get_reward((r, c)) == -1:
                    axs[1].text(c + 0.5, r + 0.5, 'X', ha='center', va='center', fontsize=20, color='red')
                    continue

                # Get the raw Q-values for the current state
                q_vals = self.agent.q_table[r, c]

                # Define a bounding box for the text to improve readability against the arrows.
                bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

                # Draw small arrows and the corresponding Q-value text for each action.
                # Up
                axs[1].text(c + 0.5, r + 0.25, f"{q_vals[Action.up.value]:.2f}", ha='center', va='center', fontsize=8, bbox=bbox_props)
                axs[1].arrow(c + 0.5, r + 0.4, 0, -0.1, head_width=0.07, color='grey', alpha=0.7)
                # Right
                axs[1].text(c + 0.75, r + 0.5, f"{q_vals[Action.right.value]:.2f}", ha='center', va='center', fontsize=8, bbox=bbox_props)
                axs[1].arrow(c + 0.6, r + 0.5, 0.1, 0, head_width=0.07, color='grey', alpha=0.7)
                # Down
                axs[1].text(c + 0.5, r + 0.75, f"{q_vals[Action.down.value]:.2f}", ha='center', va='center', fontsize=8, bbox=bbox_props)
                axs[1].arrow(c + 0.5, r + 0.6, 0, 0.1, head_width=0.07, color='grey', alpha=0.7)
                # Left
                axs[1].text(c + 0.25, r + 0.5, f"{q_vals[Action.left.value]:.2f}", ha='center', va='center', fontsize=8, bbox=bbox_props)
                axs[1].arrow(c + 0.4, r + 0.5, -0.1, 0, head_width=0.07, color='grey', alpha=0.7)

        axs[1].set_xlim(0, self.world.size)
        axs[1].set_ylim(self.world.size, 0)

        # Display the plots.
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

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

        # Call the new method to show the graphical plots.
        self.plot_results()

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
