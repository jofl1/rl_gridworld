# config.py
import numpy as np

# constants for editor
Empty = 0
Wall = 1
Start = 2
Goal = 3
Fire = 4

# Map constants to colors and symbols for drawing
Cell_style = {
    Empty: {'color': 'white', 'symbol': ''},
    Wall: {'color': 'gray', 'symbol': '■'},
    Start: {'color': 'blue', 'symbol': 'S'},
    Goal: {'color': 'green', 'symbol': 'G'},
    Fire: {'color': 'red', 'symbol': 'X'}
}

class Config:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.episodes = 100000
        self.steps = 50
        self.grid_size = 12
        self.walls = []
        self.fire_pos = []
        self.start_pos = None
        self.goal_pos = None
        
        # New behavioural configuration
        self.allow_path_crossing = False  # If True, agent can revisit states
        self.use_jofl_algorithm = True  # If True, randomly choose among similar Q-values
        self.jofl_threshold = 0.1     # Percentage threshold for considering Q-values similar