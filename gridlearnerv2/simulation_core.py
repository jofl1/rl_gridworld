# simulation_core.py
import random
import numpy as np
from config import Config

# Represents a discrete action (Up, Down, Left, Right).
class Action:
    def __init__(self, action_index: int):
        self.action_index = action_index
    def name(self) -> str: return ['Up', 'Right', 'Down', 'Left'][self.action_index]
    def get_index(self) -> int: return self.action_index
    def symbol(self) -> str: return ['↑', '→', '↓', '←'][self.action_index]
    @staticmethod
    def get_random(): return Action(random.randint(0, 3))

# Represents a single state (a cell) in the grid.
class State:
    def __init__(self, row: int, col: int, reward: int):
        self.row, self.col, self.reward = row, col, reward
        self.q_values = np.zeros(4)
    def get_reward(self) -> int: return self.reward
    def get_q_value(self, action: Action) -> float: return self.q_values[action.get_index()]
    def set_q_value(self, action: Action, q_value: float): self.q_values[action.get_index()] = q_value
    def get_max_action(self) -> Action: return Action(np.argmax(self.q_values))
    def is_terminal(self) -> bool: return self.reward != 0
    def max_future_q(self) -> float: return 0.0 if self.is_terminal() else np.max(self.q_values)
    def to_string(self) -> str: return f"({self.row}, {self.col})"

# Defines the environment (grid, rewards, and state transition logic)
# MODIFIED: Now built entirely from the Config object.
class World:
    def __init__(self, config: Config):
        self.size = config.grid_size
        self.walls = set(config.walls)
        self.config = config # Store config to access start/goal/fire info

        # Dictionary to store State objects for traversable cells only
        self.states = {}
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.walls:
                    self.states[(r, c)] = State(r, c, self.reward(r, c))

    def get_start_state(self) -> State:
        return self.states[self.config.start_pos]

    def reward(self, row: int, col: int) -> int:
        if (row, col) == self.config.goal_pos: return +1
        if (row, col) in self.config.fire_pos: return -1
        return 0

    def get_next_state(self, state: State, action: Action) -> State:
        target_row, target_col = state.row, state.col
        if action.get_index() == 0: target_row = max(0, state.row - 1)
        elif action.get_index() == 1: target_col = min(self.size - 1, state.col + 1)
        elif action.get_index() == 2: target_row = min(self.size - 1, state.row + 1)
        else: target_col = max(0, state.col - 1)
        
        if (target_row, target_col) in self.walls:
            return state # Wall - agent stays in current state
        else:
            return self.states[(target_row, target_col)]

# The learning agent. (Unchanged)
class Agent:
    def __init__(self, world: World, config: Config):
        self.world = world
        self.config = config
    def choose_action(self, state: State) -> Action:
        return Action.get_random() if random.uniform(0, 1) < self.config.exploration_rate else state.get_max_action()
    def update_q_value(self, state: State, action: Action, next_state: State):
        current_q = state.get_q_value(action)
        target_q = next_state.get_reward() + self.config.discount_factor * next_state.max_future_q()
        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        state.set_q_value(action, new_q)

# The trainer. (Unchanged)
class Trainer:
    def __init__(self, agent: Agent, world: World, config: Config):
        self.agent, self.world, self.config = agent, world, config
        self.episode_rewards, self.current_episode, self.latest_path = [], 0, None
    def run_one_episode(self):
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
        self.latest_path = (path_x, path_y)
        self.current_episode += 1
    def train(self):
        while self.current_episode < self.config.episodes:
            self.run_one_episode()
        print("\nTraining finished.")
