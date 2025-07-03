import numpy as np #type:ignore
import random
import matplotlib.pyplot as plt #type:ignore

# Stores all hyperparameters for the simulation.
class Config:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.episodes = 1000
        self.steps = 20
        self.grid_size = 4

# Represents a possible action the agent can take.
class Action:
    def __init__(self, action_index: int):
        self.action_index = action_index
        return

    def name(self):
        return ['Up', 'Right', 'Down', 'Left'][self.action_index]

    def get_index(self):
        return self.action_index

    def symbol(self):
        return ['↑', '→', '↓', '←'][self.action_index]

    @staticmethod
    def get_random():
        return Action(random.randint(0, 3))

# Represents a single state (a cell) in the grid, holding its reward and Q-values.
class State:
    def __init__(self, row: int, col: int, reward: int):
        self.row = row
        self.col = col
        self.reward = reward
        self.q_values = np.zeros(4)

    def get_reward(self):
        return self.reward

    def get_q_value(self, action: Action):
        return self.q_values[action.get_index()]

    def set_q_value(self, action: Action, q_value: float):
        self.q_values[action.get_index()] = q_value

    def get_max_action(self):
        return Action(np.argmax(self.q_values))

    def is_terminal(self):
        return self.reward != 0

    def max_future_q(self):
        return 0.0 if self.is_terminal() else np.max(self.q_values)

    def to_string(self):
        return f"({self.row} ,{self.col})"

# Defines the environment, including the grid layout, rewards, and state transitions.
class World:
    def __init__(self, config: Config):
        self.size = config.grid_size
        self.grid = []
        for row in range(self.size):
            cols = []
            for col in range(self.size):
                cols.append(State(row, col, self.reward(row, col)))
            self.grid.append(cols)

    def get_start_state(self) -> State:
        return self.grid[3][0]

    def reward(self, row: int, col: int):
        if row == 0 and col == self.size - 1:
            return +1
        elif row == 1 and col == 1:
            return -1
        return 0

    def get_next_state(self, state: State, action: Action) -> State:
        if action.get_index() == 0:
            return self.grid[max(0, state.row - 1)][state.col]
        elif action.get_index() == 1:
            return self.grid[state.row][min(self.size - 1, state.col + 1)]
        elif action.get_index() == 2:
            return self.grid[min(self.size - 1, state.row + 1)][state.col]
        return self.grid[state.row][max(0, state.col - 1)]

# Represents the learning agent that decides on actions and updates its Q-values.
class Agent:
    def __init__(self, world: World, config: Config):
        self.world = world
        self.config = config

    # Chooses an action using an epsilon-greedy policy.
    def choose_action(self, state: State):
        if random.uniform(0, 1) < self.config.exploration_rate:
            return Action.get_random()
        else:
            return state.get_max_action()

    # Updates the Q-value for a state-action pair using the Bellman equation.
    def update_q_value(self, state: State, action: Action, next_state: State):
        current_q = state.get_q_value(action)
        target_q = next_state.get_reward() + self.config.discount_factor * next_state.max_future_q()
        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        state.set_q_value(action, new_q)

# Manages the training loop over a set number of episodes.
class Trainer:
    def __init__(self, agent: Agent, world: World, config: Config):
        self.agent = agent
        self.world = world
        self.config = config
        self.episode_rewards = []

    def train(self):
        for episode in range(self.config.episodes):
            state: State = self.world.get_start_state()
            total_reward = 0
            done = False
            steps = 0
            while not done and steps < self.config.steps:
                action = self.agent.choose_action(state)
                next_state: State = self.world.get_next_state(state, action)
                reward = next_state.get_reward()
                done = next_state.is_terminal()
                self.agent.update_q_value(state, action, next_state)
                state = next_state
                total_reward += reward
                steps += 1
            self.episode_rewards.append(total_reward)
            if (episode + 1) % 1 == 0:
                avg_reward = np.mean(self.episode_rewards[-10000:])
                print(f"Episode {episode + 1}/{self.config.episodes} | Avg Reward: {avg_reward:.2f}")
        print("\nTraining finished.")

# Handles the visualisation of the final policy and Q-values.
class Visualiser:
    def __init__(self, agent: Agent, world: World, config: Config, episode_rewards: list):
        self.agent = agent
        self.world = world
        self.config = config
        self.episode_rewards = episode_rewards

    # Creates a graphical plot of the Q-values for each state-action pair.
    def plot_q_values(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Q-Values per State-Action')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

        for r in range(self.world.size):
            for c in range(self.world.size):
                state = self.world.grid[r][c]
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=0.5))

                if state.get_reward() == 1:
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20, color='green')
                    continue
                if state.get_reward() == -1:
                    ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', fontsize=20, color='red')
                    continue

                q_vals = state.q_values
                bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

                positions = {
                    'Up':    {'x': c + 0.5, 'y': r + 0.25, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.5, 'arrow_y': r + 0.4, 'dx': 0, 'dy': -0.1},
                    'Right': {'x': c + 0.75, 'y': r + 0.5, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.6, 'arrow_y': r + 0.5, 'dx': 0.1, 'dy': 0},
                    'Down':  {'x': c + 0.5, 'y': r + 0.75, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.5, 'arrow_y': r + 0.6, 'dx': 0, 'dy': 0.1},
                    'Left':  {'x': c + 0.25, 'y': r + 0.5, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.4, 'arrow_y': r + 0.5, 'dx': -0.1, 'dy': 0}
                }

                for i in range(4):
                    action = Action(i)
                    pos = positions[action.name()]
                    ax.text(pos['x'], pos['y'], f"{q_vals[i]:.2f}", ha=pos['ha'], va=pos['va'], fontsize=8, bbox=bbox_props)
                    ax.arrow(pos['arrow_x'], pos['arrow_y'], pos['dx'], pos['dy'], head_width=0.07, color='grey', alpha=0.7)

        ax.set_xlim(0, self.world.size)
        ax.set_ylim(self.world.size, 0)
        plt.tight_layout()
        plt.show()

    # Displays the final learned policy and summary statistics.
    def display_results(self):
        print("\nLearned Policy (Best move from each cell):")
        for r in range(self.world.size):
            row_str = ""
            for c in range(config.grid_size):
                state = self.world.grid[r][c]
                if state.get_reward() == 1: row_str += " G  "
                elif state.get_reward() == -1: row_str += " X  "
                else:
                    best_action = state.get_max_action()
                    row_str += f" {best_action.symbol()}  "
            print(row_str)

        final_avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = sum(r > 0 for r in self.episode_rewards) / self.config.episodes * 100
        print(f"\nFinal {config.episodes} episode average reward: {final_avg_reward:.2f}")
        print(f"Success rate (reaching goal): {success_rate:.1f}%")

        self.plot_q_values()

# Main execution block: initialises and runs the simulation.
config = Config()
world = World(config)
agent = Agent(world, config)
trainer = Trainer(agent, world, config)
trainer.train()

visualiser = Visualiser(agent, world, config, trainer.episode_rewards)
visualiser.display_results()
