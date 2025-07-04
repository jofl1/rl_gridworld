# config.py
import numpy as np

# constants for editor
Empty = 0
Wall = 1
Start = 2
Goal = 3
Fire = 4

# Map constants to colors and symbols for drawing
CELL_STYLE = {
    Empty: {'color': 'white', 'symbol': ''},
    Wall: {'color': 'gray', 'symbol': '■'},
    Start: {'color': 'blue', 'symbol': 'S'},
    Goal: {'color': 'green', 'symbol': 'G'},
    Fire: {'color': 'red', 'symbol': 'X'}
}

class Config:
    def __init__(self):
        self.learning_rate = 0.3
        self.discount_factor = 0.9
        self.exploration_rate = 0.5
        self.episodes = 100000
        self.steps = 500
        self.grid_size = 10
        self.walls = []
        self.fire_pos = []
        self.start_pos = None
        self.goal_pos = None