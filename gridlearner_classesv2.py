from matplotlib import pyplot as plt #type:ignore
from matplotlib.animation import FuncAnimation #type:ignore
from matplotlib.widgets import Button #type:ignore
import numpy as np #type:ignore
import random


# A data class holding all hyperparameters for the simulation.
# Passed into objects that require simulation parameters.
class Config:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.episodes = 10000
        self.steps = 20
        self.grid_size = 8
        
        # Define walls as a list of (row, col) tuples - empty by default
        self.walls = []
        
        # Example wall configurations (uncomment to test):
        # Vertical wall in the middle
        # self.walls = [(r, self.grid_size // 2) for r in range(1, self.grid_size - 1)]
        
        # L-shaped wall
        # self.walls = [(1, 2), (2, 2), (2, 1)]
        
        # Maze-like configuration
        self.walls = [(0, 1), (1, 1), (1, 3), (2, 3), (3, 1), (3, 2)]


# Represents a discrete action (Up, Down, Left, Right).
# Initialised with an integer index. Provides methods to get names and symbols.
class Action:
    def __init__(self, action_index: int):
        self.action_index = action_index

    # Returns the string name of the action
    def name(self) -> str:
        return ['Up', 'Right', 'Down', 'Left'][self.action_index]

    # Returns the integer index of the action
    def get_index(self) -> int:
        return self.action_index

    # Returns a symbol for the action
    def symbol(self) -> str:
        return ['↑', '→', '↓', '←'][self.action_index]

    # Factory method to generate a random action. Returns an Action instance
    @staticmethod
    def get_random():
        return Action(random.randint(0, 3))


# Represents a single state (a cell) in the grid.
# Initialised with its coordinates and reward. Holds the Q-values for that state.
class State:
    def __init__(self, row: int, col: int, reward: int):
        self.row, self.col, self.reward = row, col, reward
        self.q_values = np.zeros(4)  # Holds Q-values for actions 

    # Returns the state's intrinsic reward value.
    def get_reward(self) -> int:
        return self.reward

    # Takes an Action object and returns the corresponding Q-value (float)
    def get_q_value(self, action: Action) -> float:
        return self.q_values[action.get_index()]

    # Takes an Action and a float, updating the Q-value for that action
    def set_q_value(self, action: Action, q_value: float):
        self.q_values[action.get_index()] = q_value

    # Returns the Action with the highest Q-value from this state
    def get_max_action(self) -> Action:
        return Action(np.argmax(self.q_values))

    # Returns true if the state is a terminal one 
    def is_terminal(self) -> bool:
        return self.reward != 0

    # Returns the maximum Q-value available from this state
    def max_future_q(self) -> float:
        return 0.0 if self.is_terminal() else np.max(self.q_values)

    # Returns a string representation of the state's coordinates
    def to_string(self) -> str:
        return f"({self.row}, {self.col})"


# Defines the environment (grid, rewards, and state transition logic)
# Now supports walls while maintaining scalability
class World:
    def __init__(self, config: Config):
        self.size = config.grid_size
        self.walls = set(config.walls)  # Convert to set for O(1) lookup
        
        # Create the grid layout dynamically
        self.grid_layout = [[0 for _ in range(self.size)] for _ in range(self.size)]
        
        # Set goal at top-right (scalable)
        self.grid_layout[0][self.size - 1] = 1
        
        # Set fire (scalable) - only if the position exists in the grid
        if self.size > 1:
            self.grid_layout[7][7] = -1
        
        # Set walls
        for (r, c) in self.walls:
            if 0 <= r < self.size and 0 <= c < self.size:
                self.grid_layout[r][c] = 2
        
        # Dictionary to store State objects for traversable cells only
        self.states = {}
        
        # Create State objects only for non-wall cells
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.walls:
                    self.states[(r, c)] = State(r, c, self.reward(r, c))

    # Returns the designated starting State for each episode (scalable)
    def get_start_state(self) -> State:
        return self.states[(self.size - 1, 0)]

    # Internal method to define rewards for specific grid coordinates (scalable)
    def reward(self, row: int, col: int) -> int:
        if row == 0 and col == self.size - 1: return +1  # Goal at top-right
        if self.size > 1 and row == 1 and col == 1: return -1  # Fire at (1,1) if grid is big enough
        return 0

    # The environment's dynamics model. Takes a State and Action, returns the resulting State
    def get_next_state(self, state: State, action: Action) -> State:
        # Calculate target coordinates based on action
        target_row, target_col = state.row, state.col
        
        if action.get_index() == 0:  # Up
            target_row = max(0, state.row - 1)
        elif action.get_index() == 1:  # Right
            target_col = min(self.size - 1, state.col + 1)
        elif action.get_index() == 2:  # Down
            target_row = min(self.size - 1, state.row + 1)
        else:  # Left (action.get_index() == 3)
            target_col = max(0, state.col - 1)
        
        # Check if target is a wall
        if (target_row, target_col) in self.walls:
            # Wall - agent stays in current state
            return state
        else:
            # Traversable cell - return the State object at target coordinates
            return self.states[(target_row, target_col)]


# The learning agent. It decides on actions and updates Q-values.
# Initialised with the World and a Config object.
class Agent:
    def __init__(self, world: World, config: Config):
        self.world = world
        self.config = config

    # Implements epsilon-greedy policy. Takes a State, returns an Action
    def choose_action(self, state: State) -> Action:
        return Action.get_random() if random.uniform(0, 1) < self.config.exploration_rate else state.get_max_action()

    # Applies the Q-learning update rule. Takes state-action-next_state tuple, modifies Q-value in the state
    def update_q_value(self, state: State, action: Action, next_state: State):
        current_q = state.get_q_value(action)
        target_q = next_state.get_reward() + self.config.discount_factor * next_state.max_future_q()
        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        state.set_q_value(action, new_q)


class Trainer:
    def __init__(self, agent: Agent, world: World, config: Config):
        self.agent, self.world, self.config = agent, world, config
        self.episode_rewards = []  # Stores total reward for each episode
        self.current_episode = 0   # Add a counter for the current episode
        self.latest_path = None    # Store the path of the most recent episode

    def run_one_episode(self):
        """Runs a single episode of the simulation and stores the results."""
        state = self.world.get_start_state()
        total_reward, done, steps = 0, False, 0
        path_x, path_y = [state.col + 0.5], [state.row + 0.5]

        while not done and steps < self.config.steps:
            action = self.agent.choose_action(state)
            next_state = self.world.get_next_state(state, action)
            self.agent.update_q_value(state, action, next_state)
            state = next_state
            total_reward += state.get_reward()
            done = state.is_terminal()
            steps += 1
            path_x.append(state.col + 0.5)
            path_y.append(state.row + 0.5)
            
        self.episode_rewards.append(total_reward)
        self.latest_path = (path_x, path_y) # Store the path for the visualiser
        self.current_episode += 1

    def train(self):
        """Runs the main training loop over all episodes."""
        while self.current_episode < self.config.episodes:
            self.run_one_episode()
        print("\nTraining finished.")


# Handles the static visualisation of the final learned policy and Q-values
# Updated to display walls
class Visualiser:
    def __init__(self, agent: Agent, world: World, config: Config, episode_rewards: list):
        self.agent, self.world, self.config = agent, world, config
        self.episode_rewards = episode_rewards

    # Creates and shows a plot of the Q-values for each state-action pair.
    def plot_q_values(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Q-Values per State-Action')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

        for r in range(self.world.size):
            for c in range(self.world.size):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=0.5))
                
                # Check if it's a wall
                if (r, c) in self.world.walls:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, facecolor='gray', edgecolor='black', lw=0.5))
                    ax.text(c + 0.5, r + 0.5, '■', ha='center', va='center', fontsize=20, color='darkgray')
                    continue
                
                state = self.world.states[(r, c)]
                
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

    # Prints summary statistics and the final policy grid to the console
    def display_results(self):
        print("\nLearned Policy (Best move from each cell):")
        for r in range(self.world.size):
            row_str = ""
            for c in range(self.config.grid_size):
                if (r, c) in self.world.walls:  # Wall
                    row_str += " ■  "
                elif (r, c) in self.world.states:
                    state = self.world.states[(r, c)]
                    if state.get_reward() == 1:  # Goal
                        row_str += " G  "
                    elif state.get_reward() == -1:  # Fire
                        row_str += " X  "
                    else:  # Normal cell
                        best_action = state.get_max_action()
                        row_str += f" {best_action.symbol()}  "
            print(row_str)

        final_avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = sum(r > 0 for r in self.episode_rewards) / self.config.episodes * 100
        print(f"\nFinal {self.config.episodes} episode average reward: {final_avg_reward:.2f}")
        print(f"Success rate (reaching goal): {success_rate:.1f}%")
        self.plot_q_values()


class LiveVisualiser:
    """
    Couples with a Trainer instance to visualise the training process. only handles plotting.
    Updated to display walls.
    """
    def __init__(self, trainer: Trainer, world: World, config: Config):
        """
        Initialises the visualiser.
        Takes in the trainer (as the engine), world (for drawing), and config.
        """
        # Core Components 
        self.trainer = trainer
        self.world = world
        self.config = config

        # Plotting and Animation State 
        self.fig, self.ax_grid = plt.subplots(1, 1, figsize=(8, 8))
        self.frame_info = []         # Holds artists to be cleared each frame
        self.path_line = None        # Holds the artist for the agent's path
        self.success_text_artist = None # Holds the persistent success rate text
        self.last_success_rate = 0.0
        self.anim = None
        self.is_paused = False
        self.pause_button = None

        # Drawing Constants 
        self.arrow_base_size = 0.05
        self.arrow_size_multiplier = 0.1
        self.arrow_head_scale = 0.8
        self.transparency_base = 0.3
        self.transparency_multiplier = 0.7
        self.q_value_fontsize = 7
        self.animation_interval_ms = 1000

        # Pre-calculation 
        self.arrow_params = {
            0: {'dx': 0, 'dy': -0.15, 'x_offset': 0.5, 'y_offset': 0.35},
            1: {'dx': 0.15, 'dy': 0, 'x_offset': 0.65, 'y_offset': 0.5},
            2: {'dx': 0, 'dy': 0.15, 'x_offset': 0.5, 'y_offset': 0.65},
            3: {'dx': -0.15, 'dy': 0, 'x_offset': 0.35, 'y_offset': 0.5}
        }
        self.text_positions = {}
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                for i in range(4):
                    params = self.arrow_params[i]
                    text_x = c + params['x_offset'] + params['dx'] * 0.7
                    text_y = r + params['y_offset'] + params['dy'] * 0.7
                    self.text_positions[(r, c, i)] = (text_x, text_y)

        # Draw the static background
        self.setup_plot()

    def setup_plot(self):
        """Sets up the static elements of the plot that don't change."""
        self.fig.subplots_adjust(bottom=0.1)
        self.ax_grid.set_title('Q-Learning Progress', fontsize=14)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_xlim(0, self.world.size)
        self.ax_grid.set_ylim(self.world.size, 0)

        for r in range(self.world.size):
            for c in range(self.world.size):
                self.ax_grid.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=0.5))
                
                if (r, c) in self.world.walls:  # Wall
                    self.ax_grid.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, facecolor='gray', edgecolor='black', lw=0.5))
                    self.ax_grid.text(c + 0.5, r + 0.5, '■', ha='center', va='center', fontsize=20, color='darkgray', weight='bold')
                elif (r, c) in self.world.states:
                    state = self.world.states[(r, c)]
                    if state.get_reward() == 1:  # Goal
                        self.ax_grid.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20, color='green', weight='bold')
                    elif state.get_reward() == -1:  # Fire
                        self.ax_grid.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', fontsize=20, color='red', weight='bold')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])

    def toggle_pause(self, event):
        """Handles the pause/resume button click event."""
        if self.is_paused:
            self.anim.resume()
            self.pause_button.label.set_text('Pause')
        else:
            self.anim.pause()
            self.pause_button.label.set_text('Resume')
        self.is_paused = not self.is_paused

    def update_frame(self, frame: int) -> list:
        """
        Tells the trainer to run an episode, then redraws the simulation state.
        This method contains NO Q-learning logic itself.
        """
        # 1. Advance the simulation by telling the Trainer to run one episode
        if self.trainer.current_episode < self.config.episodes:
            self.trainer.run_one_episode()

        # 2. Cleanup all artists from the previous frame
        for info in self.frame_info:
            info.remove()
        self.frame_info.clear()

        if self.path_line:
            self.path_line.remove()
            self.path_line = None

        # 3. Redraw the path from the trainer's last-run episode
        if self.trainer.latest_path:
            path_x, path_y = self.trainer.latest_path
            if len(path_x) > 1:
                self.path_line, = self.ax_grid.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.8,
                                                      marker='o', markersize=4, markerfacecolor='red',
                                                      markeredgecolor='darkred', markeredgewidth=0.5)

        # To visualise Q-values, find the largest absolute value for symmetric normalization
        all_q_values = []
        for (r, c), state in self.world.states.items():
            if not state.is_terminal():
                all_q_values.extend(state.q_values)
        
        # Find the maximum absolute value to create a symmetric range [-max, +max]
        max_abs_q = max(abs(q) for q in all_q_values) if all_q_values else 1.0
        if max_abs_q == 0: max_abs_q = 1.0 # Avoid division by zero

        # Pre-create the bbox dict to avoid recreating it multiple times
        text_bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8)

        # Iterate through the grid and draw the Q-value arrows for each state-action pair
        for r in range(self.world.size):
            for c in range(self.world.size):
                # Skip walls
                if (r, c) in self.world.walls:
                    continue
                    
                state = self.world.states[(r, c)]
                if state.is_terminal():
                    continue

                for i in range(4):
                    q_val = state.q_values[i]

                    # This maps q_val from [-max_abs_q, +max_abs_q] to a 0-1 range
                    # where negative values are 0-0.5, 0 is 0.5, and positive are 0.5-1.
                    normalized_q = (q_val / max_abs_q) * 0.5 + 0.5
                
                    # Size is based on distance from the neutral center (0.5)
                    # abs(normalized_q - 0.5) gives a value from 0 to 0.5. Multiplying by 2 scales it to 0-1.
                    size_factor = abs(normalized_q - 0.5) * 2
                    arrow_size = self.arrow_base_size + size_factor * self.arrow_size_multiplier
                
                    # Red -> Yellow(Neutral) -> Green
                    color = plt.cm.RdYlGn(normalized_q)
                
                    # transparency can be based on the same size factor to make neutral arrows more transparent
                    transparency = self.transparency_base + size_factor * self.transparency_multiplier

                    params = self.arrow_params[i]
                    arrow = self.ax_grid.arrow(c + params['x_offset'], r + params['y_offset'],
                                           params['dx'] * size_factor, params['dy'] * size_factor, # Scale arrow length by size_factor
                                           head_width=arrow_size, head_length=arrow_size * self.arrow_head_scale,
                                           fc=color, ec=color, alpha=transparency)
                    self.frame_info.append(arrow)

                    text_x, text_y = self.text_positions[(r, c, i)]
                    text = self.ax_grid.text(text_x, text_y, f"{q_val:.1f}", ha='center', va='center',
                                         fontsize=self.q_value_fontsize, bbox=text_bbox)
                    self.frame_info.append(text)

        episode_text = self.ax_grid.text(0.02, 0.98, f"Episode: {self.trainer.current_episode}/{self.config.episodes}",
                                         transform=self.ax_grid.transAxes, ha='left', va='top',
                                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7), fontsize=10)
        self.frame_info.append(episode_text)

        # Update and display the persistent success rate text
        rewards = self.trainer.episode_rewards
        if rewards and (self.trainer.current_episode % 10 == 0 or self.trainer.current_episode == self.config.episodes):
            self.last_success_rate = sum(r > 0 for r in rewards) / len(rewards) * 100

        if self.success_text_artist is None:
            self.success_text_artist = self.ax_grid.text(0.98, 0.98, "", transform=self.ax_grid.transAxes,
                                                         ha='right', va='top',
                                                         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7),
                                                         fontsize=10)
        
        self.success_text_artist.set_text(f"Success Rate: {self.last_success_rate:.1f}%")

        return self.frame_info + [self.path_line]

    def animate_training(self):
        """Creates and returns the animation object and the button."""
        self.anim = FuncAnimation(self.fig, self.update_frame, frames=self.config.episodes,
                                  interval=self.animation_interval_ms, repeat=False, blit=False)

        button_ax = self.fig.add_axes([0.4, 0.02, 0.2, 0.05])
        self.pause_button = Button(button_ax, 'Pause', color='lightgoldenrodyellow', hovercolor='0.975')
        self.pause_button.on_clicked(self.toggle_pause)
        
        return self.anim


# Initialise the core components
config = Config()

# Example: Test with a larger grid and walls
# config.grid_size = 8
# config.walls = [(r, 4) for r in range(1, 7)] + [(3, c) for c in range(2, 6)]

world = World(config)
agent = Agent(world, config)

# Handle user choice for visualisation mode
print("Select visualisation mode:")
print("1. Live ")
print("2. Final results ")
choice = input("1 or 2: ")

if choice == "1":
    # 1. Create the core components as before
    trainer = Trainer(agent, world, config)

    # 2. Create the visualiser and PASS IT THE TRAINER
    live_visualiser = LiveVisualiser(trainer, world, config)
    
    # 3. Run the animation
    anim = live_visualiser.animate_training()
    plt.show()
else:
    # Run the standard training and show final results
    trainer = Trainer(agent, world, config)
    trainer.train()
    visualiser = Visualiser(agent, world, config, trainer.episode_rewards)
    visualiser.display_results()
