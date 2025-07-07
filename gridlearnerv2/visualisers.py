# visualisers.py
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

from config import Config
from simulation_core import Agent, World, Trainer, State


class BaseVisualiser:
    """Base class containing shared visualisation functionality."""
    
    # Visual constants
    _figure_size = (10, 10)
    _live_figure_size = (10, 8)
    _title_fontsize = 16
    _symbol_fontsize = 20
    _label_fontsize = 10
    _q_value_fontsize = 7
    
    # Arrow rendering parameters
    _arrow_base_size = 0.06
    _arrow_size_multiplier = 0.12
    _arrow_head_scale = 0.8
    _transparency_base = 0.4
    _transparency_multiplier = 0.6
    
    # Arrow direction parameters: maps action index to display properties
    _arrow_params = {
        0: {'dx': 0, 'dy': -0.15, 'x_offset': 0.5, 'y_offset': 0.35},   # Up
        1: {'dx': 0.15, 'dy': 0, 'x_offset': 0.65, 'y_offset': 0.5},    # Right
        2: {'dx': 0, 'dy': 0.15, 'x_offset': 0.5, 'y_offset': 0.65},    # Down
        3: {'dx': -0.15, 'dy': 0, 'x_offset': 0.35, 'y_offset': 0.5}    # Left
    }
    
    def __init__(self, world: World, config: Config):
        """
        Initialises base visualiser with world and config.
        
        Args:
            world (World): Environment instance
            config (Config): Configuration parameters
        """
        self.world = world
        self.config = config
        self._text_positions = self._precompute_text_positions()
    
    def _precompute_text_positions(self) -> dict[tuple[int, int, int], tuple[float, float]]:
        """
        Pre-calculates Q-value text positions for efficiency.
        
        Returns:
            dict: Maps (row, col, action_idx) to (x, y) display coordinates
        """
        positions = {}
        for r in range(self.world.size):
            for c in range(self.world.size):
                for i in range(4):
                    params = self._arrow_params[i]
                    text_x = c + params['x_offset'] + params['dx'] * 0.7
                    text_y = r + params['y_offset'] + params['dy'] * 0.7
                    positions[(r, c, i)] = (text_x, text_y)
        return positions
    
    def _setup_grid_base(self, ax):
        """
        Configures basic grid appearance.
        
        Args:
            ax: Matplotlib axes to configure
        """
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlim(0, self.world.size)
        ax.set_ylim(self.world.size, 0)
    
    def _draw_static_elements(self, ax):
        """
        Draws grid cells, walls, and special positions.
        
        Args:
            ax: Matplotlib axes to draw on
        """
        for r in range(self.world.size):
            for c in range(self.world.size):
                # Draw grid cell
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, 
                                         edgecolor='black', lw=0.5))
                
                pos = (r, c)
                if pos in self.config.world['walls']:
                    # Draw wall
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, 
                                             facecolor='gray', edgecolor='black', lw=0.5))
                    ax.text(c + 0.5, r + 0.5, '■', ha='center', va='center', 
                           fontsize=self._symbol_fontsize, color='darkgray', weight='bold')
                elif pos == self.config.world['start_pos']:
                    # Draw start position
                    ax.text(c + 0.1, r + 0.1, 'S', ha='left', va='top', 
                           fontsize=self._label_fontsize, color='blue', weight='bold')
                elif pos in self.world.states:
                    state = self.world.states[pos]
                    if state.get_reward() == 1:
                        # Draw goal
                        ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', 
                               fontsize=self._symbol_fontsize, color='green', weight='bold')
                    elif state.get_reward() == -1:
                        # Draw hazard
                        ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', 
                               fontsize=self._symbol_fontsize, color='red', weight='bold')
    
    def _calculate_q_normalisation(self) -> float:
        """
        Calculates normalisation factor for Q-value visualisation.
        
        Returns:
            float: Maximum absolute Q-value across all non-terminal states
        """
        all_q_values = [
            q for state in self.world.states.values() 
            if not state.is_terminal() 
            for q in state.q_values
        ]
        max_abs_q = max(abs(q) for q in all_q_values) if all_q_values else 1.0
        return max(max_abs_q, 1.0)  # Avoid division by zero
    
    def _draw_q_arrows(self, ax, state: State, row: int, col: int, 
                      max_abs_q: float, frame_info: list = None):
        """
        Draws Q-value arrows and text for a single state.
        
        Args:
            ax: Matplotlib axes
            state (State): State to visualise
            row (int): Row position
            col (int): Column position
            max_abs_q (float): Normalisation factor
            frame_info (list, optional): List to store arrow objects for animation
        """
        text_bbox = dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.8)
        
        for i in range(4):
            q_val = state.q_values[i]
            
            # Normalise Q-value to [0, 1] range
            normalized_q = (q_val / max_abs_q) * 0.5 + 0.5
            size_factor = abs(normalized_q - 0.5) * 2
            
            # Calculate arrow properties
            arrow_size = self._arrow_base_size + size_factor * self._arrow_size_multiplier
            color = plt.cm.RdYlGn(normalized_q)
            transparency = self._transparency_base + size_factor * self._transparency_multiplier
            
            # Draw arrow
            params = self._arrow_params[i]
            arrow = ax.arrow(
                col + params['x_offset'], row + params['y_offset'],
                params['dx'] * size_factor, params['dy'] * size_factor,
                head_width=arrow_size, head_length=arrow_size * self._arrow_head_scale,
                fc=color, ec=color, alpha=transparency
            )
            
            if q_val < 0.01:
                q_val_color = 'yellow'
            else:
                q_val_color = 'blue'
            
            # Draw Q-value text
            text_x, text_y = self._text_positions[(row,col,i)]
            text = ax.text(text_x, text_y, f"{q_val:.2f}", 
                          ha='center', va='center', color = q_val_color,
                          fontsize=self._q_value_fontsize, bbox=text_bbox)
            
            # Store references if animating
            if frame_info is not None:
                frame_info.extend([arrow, text])


class Visualiser(BaseVisualiser):
    """Static visualiser for displaying final Q-values and policy."""
    
    def __init__(self, agent: Agent, world: World, config: Config, episode_rewards: list[int]):
        """
        Initialises static visualiser.
        
        Args:
            agent (Agent): Trained agent
            world (World): Environment
            config (Config): Configuration
            episode_rewards (list[int]): Rewards from all training episodes
        """
        super().__init__(world, config)
        self.agent = agent
        self.episode_rewards = episode_rewards
        self.config = config
        self.world = world
    
    def plot_q_values(self):
        """Creates and displays the Q-value visualisation plot."""
        fig, ax = plt.subplots(figsize=self._figure_size)
        ax.set_title('Q-Values per State-Action', fontsize=self._title_fontsize, fontweight='bold')
        
        # Set up grid
        self._setup_grid_base(ax)
        self._draw_static_elements(ax)
        
        # Calculate normalisation
        max_abs_q = self._calculate_q_normalisation()
        
        # Draw Q-values for each state
        for pos, state in self.world.states.items():
            if not state.is_terminal() and pos not in self.config.world['walls']:
                self._draw_q_arrows(ax, state, pos[0], pos[1], max_abs_q)
        
        plt.tight_layout()
        plt.show()
    
    def _print_policy(self):
        """Prints the learned policy as a text grid."""
        print("\nLearned Policy (Best move from each cell):")
        for r in range(self.config.world['grid_size']):
            row_str = ""
            for c in range(self.config.world['grid_size']):
                pos = (r, c)
                if pos in self.config.world['walls']:
                    row_str += " ■  "
                elif pos in self.world.states:
                    state = self.world.states[pos]
                    if state.get_reward() == 1:
                        row_str += " G  "
                    elif state.get_reward() == -1:
                        row_str += " X  "
                    else:
                        row_str += f" {state.get_max_action().symbol()}  "
            print(row_str)
    
    def _print_statistics(self):
        """Prints training statistics."""
        if not self.episode_rewards:
            return
        
        # Calculate metrics
        final_avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = sum(r > 0 for r in self.episode_rewards) / len(self.episode_rewards) * 100
        
        print(f"\nFinal {len(self.episode_rewards)} episode average reward: {final_avg_reward:.2f}")
        print(f"Success rate (reaching goal): {success_rate:.1f}%")
    
    def display_results(self):
        """Shows complete results including policy, statistics, and Q-value plot."""
        self._print_policy()
        self._print_statistics()
        self.plot_q_values()


class LiveVisualiser(BaseVisualiser):
    """Animated visualiser showing Q-learning progress in real-time."""
    
    _animation_interval_ms = 1
    _update_frequency = 10  # Update success rate every N episodes
    
    def __init__(self, trainer: Trainer, world: World, config: Config):
        """
        Initialises live visualiser.
        
        Args:
            trainer (Trainer): Trainer instance to monitor
            world (World): Environment
            config (Config): Configuration
        """
        super().__init__(world, config)
        self.trainer = trainer
        
        # Animation state
        self.frame_info = []          # list: Temporary arrow/text objects
        self.path_line = None         # Line2D: Episode path visualisation
        self.last_success_rate = 0.0  # float: Cached success rate
        self.is_paused = False        # bool: Pause state
        
        # UI elements
        self.fig = None
        self.ax_grid = None
        self.anim = None
        self.pause_button = None
        self.episode_text = None
        self.success_text = None
        
        self._setup_plot()
    
    def _setup_plot(self):
        """Creates the figure and initial plot elements."""
        self.fig, self.ax_grid = plt.subplots(1, 1, figsize=self._live_figure_size)
        self.fig.subplots_adjust(bottom=0.15, top=0.9)
        
        # Configure grid
        self.ax_grid.set_title('Q-Learning Progress', fontsize=14, pad=20)
        self._setup_grid_base(self.ax_grid)
        self._draw_static_elements(self.ax_grid)
        
        # Add text displays
        self.episode_text = self.fig.text(0.02, 0.95, '', fontsize=12, fontweight='bold')
        self.success_text = self.fig.text(0.98, 0.95, '', ha='right', fontsize=12, fontweight='bold')
    
    def _draw_episode_path(self):
        """Draws the path taken in the latest episode."""
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        
        if self.trainer.latest_path:
            path_x, path_y = self.trainer.latest_path
            if len(path_x) > 1:
                self.path_line, = self.ax_grid.plot(
                    path_x, path_y, 'b-', 
                    linewidth=2, alpha=0.8, 
                    marker='o', markersize=4
                )
    
    def _update_statistics(self):
        """Updates displayed statistics."""
        self.episode_text.set_text(
            f"Episode: {self.trainer.current_episode}/{self.config.episodes}"
        )
        
        # Update success rate periodically
        rewards = self.trainer.episode_rewards
        if rewards and (self.trainer.current_episode % self._update_frequency == 0 or 
                       self.trainer.current_episode == self.config.episodes):
            self.last_success_rate = sum(r > 0 for r in rewards) / len(rewards) * 100
        
        self.success_text.set_text(f"Success Rate: {self.last_success_rate:.1f}%")
    
    def update_frame(self, frame: int) -> list:
        """
        Updates the visualisation for one animation frame.
        
        Args:
            frame (int): Frame number
            
        Returns:
            list: Updated artist objects for blitting
        """
        # Run training step if not complete
        if self.trainer.current_episode < self.config.episodes:
            self.trainer.run_one_episode()
        
        # Clear previous frame elements
        for element in self.frame_info:
            element.remove()
        self.frame_info.clear()
        
        # Draw episode path
        self._draw_episode_path()
        
        # Calculate normalisation
        max_abs_q = self._calculate_q_normalisation()
        
        # Draw Q-values for all states
        for pos, state in self.world.states.items():
            if not state.is_terminal() and pos not in self.config.world['walls']:
                self._draw_q_arrows(self.ax_grid, state, pos[0], pos[1], 
                                  max_abs_q, self.frame_info)
        
        # Update statistics
        self._update_statistics()
        
        # Return all artists that need updating
        artists = self.frame_info.copy()
        if self.path_line:
            artists.append(self.path_line)
        artists.extend([self.episode_text, self.success_text])
        
        return artists
    
    def toggle_pause(self, event):
        """
        Toggles animation pause state.
        
        Args:
            event: Button click event
        """
        if self.is_paused:
            self.anim.resume()
            self.pause_button.label.set_text('Pause')
        else:
            self.anim.pause()
            self.pause_button.label.set_text('Resume')
        self.is_paused = not self.is_paused
    
    def animate_training(self) -> FuncAnimation:
        """
        Starts the training animation.
        
        Returns:
            FuncAnimation: Animation object
        """
        # Create animation
        self.anim = FuncAnimation(
            self.fig, self.update_frame, 
            frames=self.config.episodes + 1,
            interval=self._animation_interval_ms,
            repeat=False, 
            blit=False
        )
        
        # Add pause button
        button_ax = self.fig.add_axes([0.45, 0.05, 0.1, 0.05])
        self.pause_button = Button(button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)
        
        return self.anim