# editor.py
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.gridspec import GridSpec
import numpy as np

# Import the new TitleCase constant names
from config import Config, Empty, Wall, Start, Goal, Fire, CELL_STYLE
from simulation_core import World, Agent, Trainer
from visualisers import Visualiser, LiveVisualiser


class GridEditor:
    """
    An interactive matplotlib window to let the user define the game grid.
    The user can "paint" walls, start, goal, and fire squares.
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), Empty, dtype=int)
        
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.suptitle("Gridworld Editor", fontsize=16, fontweight='bold')
        
        # Use GridSpec for layout
        gs = GridSpec(1, 2, width_ratios=[3, 1], figure=self.fig)
        
        # Grid Axes
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax.set_title("Design your map, then press 'Run'")
        self.ax.set_aspect('equal')

        self.cell_patches = {}
        self.ax.set_xticks(np.arange(grid_size + 1))
        self.ax.set_yticks(np.arange(grid_size + 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_xlim(0, grid_size)
        self.ax.set_ylim(grid_size, 0)

        for r in range(grid_size):
            for c in range(grid_size):
                rect = plt.Rectangle((c, r), 1, 1, facecolor='white', edgecolor='grey')
                symbol = self.ax.text(c + 0.5, r + 0.5, '', ha='center', va='center', fontsize=16, color='black')
                self.ax.add_patch(rect)
                self.cell_patches[(r, c)] = {'rect': rect, 'symbol': symbol}

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Controls Frame
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        controls_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1])

        # Radio buttons for selecting tool
        rax = self.fig.add_subplot(controls_gs[0, 0])
        rax.set_title("Tools")
        self.radio = RadioButtons(rax, ('Empty', 'Wall', 'Start', 'Goal', 'Fire'))

        # Action buttons
        buttons_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=controls_gs[1, 0])
        run_live_ax = self.fig.add_subplot(buttons_gs[0, 0])
        self.run_live_button = Button(run_live_ax, 'Run Live')
        self.run_live_button.on_clicked(lambda event: self.run_simulation(live_mode=True))

        run_final_ax = self.fig.add_subplot(buttons_gs[1, 0])
        self.run_final_button = Button(run_final_ax, 'Show Final')
        self.run_final_button.on_clicked(lambda event: self.run_simulation(live_mode=False))

        reset_ax = self.fig.add_subplot(buttons_gs[2, 0])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_grid)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])

    def on_click(self, event):
        """Handles a mouse click on the grid."""
        if not event.inaxes == self.ax: return
        c, r = int(event.xdata), int(event.ydata)

        # Use the new TitleCase constant names
        tool_labels = {'Empty': Empty, 'Wall': Wall, 'Start': Start, 'Goal': Goal, 'Fire': Fire}
        paint_mode = tool_labels[self.radio.value_selected]

        if paint_mode in [Start, Goal]:
            current_pos = np.where(self.grid == paint_mode)
            if len(current_pos[0]) > 0:
                self.grid[current_pos[0][0], current_pos[1][0]] = Empty

        self.grid[r, c] = paint_mode
        self.draw_grid()

    def draw_grid(self):
        """Redraws the entire grid based on the internal state."""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_type = self.grid[r, c]
                style = CELL_STYLE[cell_type]
                self.cell_patches[(r, c)]['rect'].set_facecolor(style['color'])
                self.cell_patches[(r, c)]['symbol'].set_text(style['symbol'])
        self.fig.canvas.draw_idle()

    def reset_grid(self, event):
        """Resets the grid to be completely empty."""
        # Use the new 'Empty' constant
        self.grid.fill(Empty)
        self.draw_grid()

    def run_simulation(self, live_mode):
        """Parses the grid, creates a Config, and starts the Q-learning simulation."""
        config = Config()
        config.grid_size = self.grid_size
        
        # Use the new TitleCase constant names for parsing
        walls = np.where(self.grid == Wall)
        fire_pos = np.where(self.grid == Fire)
        start_pos = np.where(self.grid == Start)
        goal_pos = np.where(self.grid == Goal)
        
        if len(start_pos[0]) != 1 or len(goal_pos[0]) != 1:
            self.ax.set_title("ERROR: Please define exactly one Start (S) and one Goal (G).", color='red')
            self.fig.canvas.draw_idle()
            return
            
        config.walls = list(zip(walls[0], walls[1]))
        config.fire_pos = list(zip(fire_pos[0], fire_pos[1]))
        config.start_pos = (start_pos[0][0], start_pos[1][0])
        config.goal_pos = (goal_pos[0][0], goal_pos[1][0])

        plt.close(self.fig)
        
        world = World(config)
        agent = Agent(world, config)
        trainer = Trainer(agent, world, config)

        if live_mode:
            live_visualiser = LiveVisualiser(trainer, world, config)
            anim = live_visualiser.animate_training()
            plt.show()
        else:
            trainer.train()
            visualiser = Visualiser(agent, world, config, trainer.episode_rewards)
            visualiser.display_results()
