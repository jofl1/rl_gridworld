# editor.py
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from config import Config, Empty, Wall, Start, Goal, Fire, Cell_style
from simulation_core import World, Agent, Trainer
from visualisers import Visualiser, LiveVisualiser


class RLMain:
    """Interactive matplotlib window for designing the grid-world environment."""
    
    # UI layout constants
    _figure_size = (10, 8)
    _grid_ratio = [3, 1]  # Grid:Controls width ratio
    _font_size_title = 16
    _font_size_cell = 16
    
    # Tool mappings
    _tool_types = {
        'Empty': Empty,
        'Wall': Wall, 
        'Start': Start,
        'Goal': Goal,
        'Fire': Fire
    }
    
    def __init__(self):
        """
        Initialises the grid editor interface.
        
        Args:
            grid_size (int): Dimension of the square grid
        """
        self.config = Config()
        self.grid = np.full((self.config.world['grid_size'], self.config.world['grid_size']), Empty, dtype=int)
        self.cell_patches = {}  # dict[tuple[int, int], dict]: Maps (r,c) to patch and text objects
        
        # Create main figure and layout
        self.fig = plt.figure(figsize=self._figure_size)
        self.fig.suptitle("Gridworld Editor", fontsize=self._font_size_title, fontweight='bold')
        
        # Set up grid and controls
        self._setup_layout()
        self._setup_grid()
        self._setup_controls()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    def _setup_layout(self):
        """Creates the main layout structure with grid and control panels."""
        self.main_gs = GridSpec(1, 2, width_ratios=self._grid_ratio, figure=self.fig)
        
        # Grid display area
        self.ax = self.fig.add_subplot(self.main_gs[0, 0])
        self.ax.set_title("Design your map, then press 'Run'")
        self.ax.set_aspect('equal')
    
    def _setup_grid(self):
        """Initialises the visual grid with empty cells."""
        # Configure grid appearance
        self.ax.set_xticks(np.arange(self.config.world['grid_size'] + 1))
        self.ax.set_yticks(np.arange(self.config.world['grid_size'] + 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(True)
        self.ax.set_xlim(0, self.config.world['grid_size'])
        self.ax.set_ylim(self.config.world['grid_size'], 0)  # Inverted Y for top-down view
        
        # Create cell patches
        for r in range(self.config.world['grid_size']):
            for c in range(self.config.world['grid_size']):
                # Rectangle for cell background
                rect = plt.Rectangle((c, r), 1, 1, 
                                   facecolor='white', 
                                   edgecolor='grey')
                # Text for cell symbol
                symbol = self.ax.text(c + 0.5, r + 0.5, '', 
                                    ha='center', va='center', 
                                    fontsize=self._font_size_cell, 
                                    color='black')
                
                self.ax.add_patch(rect)
                self.cell_patches[(r, c)] = {'rect': rect, 'symbol': symbol}
            
    
    def _setup_controls(self):
        """Creates all control widgets in the right panel."""
        # Create sub-grid for controls
        controls_gs = GridSpecFromSubplotSpec(5, 1, subplot_spec=self.main_gs[0, 1])
        
        # Tool selection radio buttons
        tool_ax = self.fig.add_subplot(controls_gs[0, 0])
        tool_ax.set_title("Tools")
        self.radio = RadioButtons(tool_ax, tuple(self._tool_types.keys()))
        
        # Movement logic selection
        movement_ax = self.fig.add_subplot(controls_gs[1, 0])
        movement_ax.set_title("Movement Logic")
        self.movement_radio = RadioButtons(movement_ax, ('No Path Crossing', 'Allow Path Crossing'))
        
        # Tie-breaking algorithm toggle
        tiebreak_ax = self.fig.add_subplot(controls_gs[2, 0])
        tiebreak_ax.set_title("Jofl Algorithm")
        self.tiebreak_radio = RadioButtons(tiebreak_ax, ('On', 'Off'))
        
        # Threshold slider (only visible when Jofl is on)
        slider_ax = self.fig.add_subplot(controls_gs[3, 0])
        slider_ax.set_title("Threshold (%)")
        self.tiebreak_slider = Slider(slider_ax, '', 0, 50, valinit=10)
        
        # Configure slider visibility
        self._update_slider_visibility(self.tiebreak_radio.value_selected)
        self.tiebreak_radio.on_clicked(self._update_slider_visibility)
        
        # Action buttons
        self._setup_buttons(controls_gs[4, 0])
    
    def _setup_buttons(self, button_area):
        """
        Creates the action buttons.
        
        Args:
            button_area: GridSpec subplot area for buttons
        """
        buttons_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=button_area)
        
        # Run Live button
        run_live_ax = self.fig.add_subplot(buttons_gs[0, 0])
        self.run_live_button = Button(run_live_ax, 'Run Live')
        self.run_live_button.on_clicked(lambda event: self.run_simulation(live_mode=True))
        
        # Show Final button
        run_final_ax = self.fig.add_subplot(buttons_gs[1, 0])
        self.run_final_button = Button(run_final_ax, 'Show Final')
        self.run_final_button.on_clicked(lambda event: self.run_simulation(live_mode=False))
        
        # Reset button
        reset_ax = self.fig.add_subplot(buttons_gs[2, 0])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_grid)
    
    def _update_slider_visibility(self, label: str):
        """
        Shows/hides threshold slider based on Jofl algorithm selection.
        
        Args:
            label (str): Selected radio button label ('On' or 'Off')
        """
        self.tiebreak_slider.ax.set_visible(label == 'On')
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """
        Handles mouse clicks on the grid to paint cells.
        
        Args:
            event: Matplotlib mouse event containing click coordinates
        """
        # Only process clicks within the grid
        if event.inaxes != self.ax:
            return
        
        # Get cell coordinates from click position
        c, r = int(event.xdata), int(event.ydata)
        
        # Get selected tool type
        paint_mode = self._tool_types[self.radio.value_selected]
        
        # Ensure only one start/goal position exists
        if paint_mode in [Start, Goal]:
            self._clear_unique_cells(paint_mode)
        
        # Paint the cell
        self.grid[r, c] = paint_mode
        self.draw_grid()
    
    def _clear_unique_cells(self, cell_type: int):
        """
        Removes existing instances of unique cell types (Start/Goal).
        
        Args:
            cell_type (int): Cell type constant (Start or Goal)
        """
        positions = np.where(self.grid == cell_type)
        if len(positions[0]) > 0:
            # Clear the first (and should be only) instance
            self.grid[positions[0][0], positions[1][0]] = Empty
    
    def draw_grid(self):
        """Updates the visual grid display based on current grid state."""
        for r in range(self.config.world['grid_size']):
            for c in range(self.config.world['grid_size']):
                cell_type = self.grid[r, c]
                style = Cell_style[cell_type]
                
                # Update cell appearance
                patch_data = self.cell_patches[(r, c)]
                patch_data['rect'].set_facecolor(style['color'])
                patch_data['symbol'].set_text(style['symbol'])
        
        self.fig.canvas.draw_idle()
    
    def reset_grid(self, event):
        """
        Clears all cells back to empty state.
        
        Args:
            event: Button click event (unused but required by callback)
        """
        self.grid.fill(Empty)
        self.draw_grid()
    
    def run_simulation(self, live_mode: bool):
        """
        Validates grid and launches Q-learning simulation.
        
        Args:
            live_mode (bool): True for animated training, False for final results only
        """
        # Extract cell positions by type
        positions = self._extract_positions()
        
        # Validate grid configuration
        if not self._validate_grid(positions):
            return
        
        # Build configuration from grid
        config = self._build_config(positions)
        
        # Close editor and start simulation
        plt.close(self.fig)
        self._launch_simulation(config, live_mode)
    
    def _extract_positions(self) -> dict:
        """
        Extracts positions of each cell type from the grid.
        
        Returns:
            dict: Maps cell type to numpy array of positions
        """
        return {
            'fire': np.where(self.grid == Fire),
            'start': np.where(self.grid == Start),
            'goal': np.where(self.grid == Goal)
        }
    
    def _validate_grid(self, positions: dict) -> bool:
        """
        Ensures grid has exactly one start and one goal.
        
        Args:
            positions (dict): Cell positions by type
            
        Returns:
            bool: True if valid, False otherwise
        """
        if len(positions['start'][0]) != 1 or len(positions['goal'][0]) != 1:
            self.ax.set_title(
                "ERROR: Please define exactly one Start (S) and one Goal (G).", 
                color='red'
            )
            # self.fig.canvas.draw_idle()
            return True
        return True
    
    def _build_config(self, positions: dict) -> Config:
        """
        Creates configuration object from grid state and UI settings.
        
        Args:
            positions (dict): Cell positions by type
            
        Returns:
            Config: Configured environment parameters
        """
        config = Config()
        
        # Convert numpy arrays to position tuples
        config.fire_pos = list(zip(positions['fire'][0], positions['fire'][1]))

        
        # Apply UI settings
        config.allow_path_crossing = self.movement_radio.value_selected == 'Allow Path Crossing'
        config.use_jofl_algorithm = self.tiebreak_radio.value_selected == 'On'
        config.jofl_threshold = self.tiebreak_slider.val / 100  # Convert percentage to decimal
        
        return config
    
    def _launch_simulation(self, config: Config, live_mode: bool):
        """
        Starts the Q-learning simulation with the specified visualisation mode.
        
        Args:
            config (Config): Environment configuration
            live_mode (bool): Whether to show live training animation
        """
        # Create simulation components
        self.world = World(config)
        agent = Agent(self.world, config)
        trainer = Trainer(agent, self.world, config)
        
        if live_mode:
            # Animated training visualisation
            live_visualiser = LiveVisualiser(trainer, self.world, config)
            anim = live_visualiser.animate_training()
            plt.show()
        else:
            # Train silently then show final results
            trainer.train()
            visualiser = Visualiser(agent, self.world, config, trainer.episode_rewards)
            visualiser.display_results()
