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
        
        # New behavioural configuration
        self.allow_path_crossing = False  # If True, agent can revisit states
        self.use_jofl_algorithm = True  # If True, randomly choose among similar Q-values
        self.jofl_threshold = 0.1     # Percentage threshold for considering Q-values similar
        
        self.block_world1 = {
            'name':'Block world 1',
            'grid_size': 12,
            'walls': [(2,2), (3,2), (4,2), (6,4),(7,4),(9,5),(9,6),(9,7),(9,8),(4,6),(4,7),(5,6),(6,6),(6,9),(6,10),(6,11),(0,9),(1,9),(2,9)],
            'start_pos': (11,0),
            'goal_pos':(0,11)
        }
        
        self.world = self.block_world1
        
        self.learning_rate = 0.9
        self.discount_factor = 0.99
        self.exploration_rate = 0.1
        self.episodes = 2000
        self.steps = self.world['grid_size'] * 6
        self.fire_pos = []
