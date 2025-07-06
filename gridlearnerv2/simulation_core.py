# simulation_core.py
import random
import numpy as np
from config import Config

# used underscres to show private members.
class Action:
    """Encapsulates a discrete action in the four-directional movement space."""
    
    # Class constants for action properties
    _names = ['Up', 'Right', 'Down', 'Left']
    _symbols = ['↑', '→', '↓', '←']
    _deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (row_change, col_change) tuples
    
    def __init__(self, action_index: int):
        """
        Initialises an action with its corresponding index.
        
        Args:
            action_index (int): Integer in range [0, 3] representing movement direction
        """
        self.action_index = action_index
    
    def name(self) -> str:
        """
        Returns human-readable name of the action.
        
        Returns:
            str: Direction name ('Up', 'Right', 'Down', or 'Left')
        """
        return self._names[self.action_index]
    
    def get_index(self) -> int:
        """
        Retrieves the numeric identifier for array indexing.
        
        Returns:
            int: Action index (0-3) for Q-table lookups
        """
        return self.action_index
    
    def symbol(self) -> str:
        """
        Provides Unicode arrow for visualisation.
        
        Returns:
            str: Arrow character (↑, →, ↓, or ←)
        """
        return self._symbols[self.action_index]
    
    def get_delta(self) -> tuple[int, int]:
        """
        Returns movement changes for this action.
        
        Returns:
            tuple[int, int]: (row_change, col_change) where changes are -1, 0, or 1
        """
        return self._deltas[self.action_index]
    
    @staticmethod
    def get_random():
        """
        Factory method for generating uniformly random actions.
        
        Returns:
            Action: New Action instance with random direction
        """
        return Action(random.randint(0, 3))


class State:
    """Represents a single cell in the grid-world with associated Q-values."""
    
    def __init__(self, row: int, col: int, reward: int):
        """
        Initialises a state with position and immediate reward.
        
        Args:
            row (int): Vertical grid coordinate (0-based)
            col (int): Horizontal grid coordinate (0-based)
            reward (int): Immediate reward for entering this state (-1, 0, or +1)
        """
        self.row, self.col, self.reward = row, col, reward
        self.q_values = np.zeros(4)  # Float array indexed by action
        self.valid_actions = []      # List[Action] populated by World
    
    @property
    def position(self) -> tuple[int, int]:
        """
        Returns state position as a tuple.
        
        Returns:
            tuple[int, int]: (row, col) coordinates
        """
        return (self.row, self.col)
    
    def get_valid_actions(self) -> list[Action]:
        """
        Returns feasible actions from this state.
        
        Returns:
            list[Action]: Actions that lead to traversable neighbouring states
        """
        return self.valid_actions
    
    def get_reward(self) -> int:
        """
        Retrieves the immediate reward signal.
        
        Returns:
            int: Reward value (+1 for goal, -1 for hazards, 0 otherwise)
        """
        return self.reward
    
    def get_q_value(self, action: Action) -> float:
        """
        Looks up current Q-value estimate for a state-action pair.
        
        Args:
            action (Action): Action to query
            
        Returns:
            float: Current Q(s,a) estimate
        """
        return self.q_values[action.get_index()]
    
    def set_q_value(self, action: Action, q_value: float):
        """
        Updates the Q-value for a specific action.
        
        Args:
            action (Action): Action whose value is being updated
            q_value (float): New Q-value estimate
        """
        self.q_values[action.get_index()] = q_value
    
    def get_max_action(self) -> Action:
        """
        Implements greedy action selection with random tie-breaking.
        
        Returns:
            Action: Optimal action according to current Q-values
        """
        if not self.valid_actions:
            return Action(0)  # Fallback for edge cases
        
        # Build list of (action, q_value) tuples for valid actions
        q_values_valid = [(a, self.get_q_value(a)) for a in self.valid_actions]
        
        # Find maximum Q-value
        max_q = max(q for _, q in q_values_valid)
        
        # Collect all actions achieving this maximum
        max_actions = [a for a, q in q_values_valid if abs(q - max_q) < 1e-9]
        
        # Random selection amongst optimal actions
        return random.choice(max_actions)
    
    def is_terminal(self) -> bool:
        """
        Checks if this state ends an episode.
        
        Returns:
            bool: True if state has non-zero reward (goal or hazard)
        """
        return self.reward != 0
    
    def max_future_q(self) -> float:
        """
        Computes maximum expected future return from this state.
        
        Returns:
            float: max Q(s',a) for all valid actions a from this state s'
        """
        if self.is_terminal() or not self.valid_actions:
            return 0.0
        
        # Return maximum Q-value across valid actions only
        return max(self.q_values[a.get_index()] for a in self.valid_actions)
    
    def __str__(self) -> str:
        """
        String representation for debugging.
        
        Returns:
            str: Coordinate tuple as string "(row, col)"
        """
        return f"({self.row}, {self.col})"
    
    def to_string(self) -> str:
        """
        Kept for backward compatibility with GUI.
        
        Returns:
            str: Same as __str__
        """
        return str(self)


class World:
    """Defines the grid-world environment with obstacles and rewards."""
    
    def __init__(self, config: Config):
        """
        Constructs environment from configuration parameters.
        
        Args:
            config (Config): Object containing grid_size, walls, goal_pos, etc.
        """
        self.size = config.grid_size          # int: Grid dimension
        self.walls = set(config.walls)       # set[tuple[int, int]]: Wall positions
        self.config = config                  # Config: Full configuration
        self.states = {}                      # dict[tuple[int, int], State]: Position to state mapping
        
        # Build state space - only traversable cells become States
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self.walls:
                    self.states[(r, c)] = State(r, c, self._get_reward(r, c))
        
        # Pre-compute valid actions for each state
        for state in self.states.values():
            state.valid_actions = self._compute_valid_actions(state)
    
    def _get_reward(self, row: int, col: int) -> int:
        """
        Returns reward for a position.
        
        Args:
            row (int): Row coordinate
            col (int): Column coordinate
            
        Returns:
            int: +1 for goal, -1 for fire/hazard, 0 for empty
        """
        if (row, col) == self.config.goal_pos:
            return +1
        if (row, col) in self.config.fire_pos:
            return -1
        return 0
    
    def _compute_valid_actions(self, state: State) -> list[Action]:
        """
        Determines which actions are valid from a given state.
        
        Args:
            state (State): State to check actions from
            
        Returns:
            list[Action]: Actions that don't hit walls or boundaries
        """
        valid = []
        for action_idx in range(4):
            action = Action(action_idx)
            dr, dc = action.get_delta()  # Get row/col changes
            next_pos = (state.row + dr, state.col + dc)
            
            if self._is_valid_position(next_pos):
                valid.append(action)
        return valid
    
    def _is_valid_position(self, pos: tuple[int, int]) -> bool:
        """
        Checks if position is within bounds and not a wall.
        
        Args:
            pos (tuple[int, int]): (row, col) position to validate
            
        Returns:
            bool: True if position is traversable
        """
        r, c = pos
        return (0 <= r < self.size and 
                0 <= c < self.size and 
                pos not in self.walls)
    
    def get_start_state(self) -> State:
        """
        Returns the initial state for episodes.
        
        Returns:
            State: Starting position state object
        """
        return self.states[self.config.start_pos]
    
    def reward(self, row: int, col: int) -> int:
        """
        Kept for backward compatibility with GUI.
        
        Args:
            row (int): Row coordinate
            col (int): Column coordinate
            
        Returns:
            int: Reward at position
        """
        return self._get_reward(row, col)
    
    def get_next_state(self, state: State, action: Action) -> State:
        """
        Returns resulting state after action execution.
        
        Args:
            state (State): Current state
            action (Action): Action to execute
            
        Returns:
            State: Next state (same as current if movement blocked)
        """
        dr, dc = action.get_delta()
        next_pos = (state.row + dr, state.col + dc)
        
        # Return next state if valid, otherwise current state
        return self.states.get(next_pos, state)


class Agent:
    """Implements Q-learning algorithm with epsilon-greedy exploration."""
    
    def __init__(self, world: World, config: Config):
        """
        Initialises agent with environment reference and hyperparameters.
        
        Args:
            world (World): Environment instance for state lookups
            config (Config): Configuration with learning_rate, exploration_rate, etc.
        """
        self.world = world
        self.config = config
    
    def choose_action(self, state: State, action_mask: list[Action] = None) -> Action:
        """
        Selects action using epsilon-greedy with configurable tie-breaking.
        
        Args:
            state (State): Current state
            action_mask (list[Action], optional): Subset of actions to consider
            
        Returns:
            Action: Selected action
            
        Raises:
            ValueError: If no actions available
        """
        # Use provided actions or all valid actions from state
        available = action_mask or state.get_valid_actions()
        
        if not available:
            raise ValueError(f"No available actions in state {state}")
        
        # Exploration: random action with probability epsilon
        if random.uniform(0, 1) < self.config.exploration_rate:
            return random.choice(available)
        
        # Exploitation: choose best action(s)
        q_values = [(a, state.get_q_value(a)) for a in available]
        max_q = max(q for _, q in q_values)
        
        if self.config.use_jofl_algorithm:
            # Soft tie-breaking: include near-optimal actions
            threshold = abs(max_q * self.config.jofl_threshold)
            candidates = [a for a, q in q_values if max_q - q <= threshold]
        else:
            # Strict tie-breaking: only exactly optimal actions
            candidates = [a for a, q in q_values if abs(q - max_q) < 1e-9]
        
        # Random choice for JOFL, first action otherwise
        return random.choice(candidates) if self.config.use_jofl_algorithm else candidates[0]
    
    def update_q_value(self, state: State, action: Action, next_state: State):
        """
        Performs temporal difference update using Q-learning rule.
        
        Updates: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state (State): Current state s
            action (Action): Executed action a
            next_state (State): Resulting state s'
        """
        current_q = state.get_q_value(action)
        
        # Calculate target value: immediate reward + discounted future value
        target_q = (next_state.get_reward() + 
                   self.config.discount_factor * next_state.max_future_q())
        
        # Update towards target with learning rate
        new_q = current_q + self.config.learning_rate * (target_q - current_q)
        state.set_q_value(action, new_q)


class Trainer:
    """Orchestrates the training loop and episode management."""
    
    def __init__(self, agent: Agent, world: World, config: Config):
        """
        Initialises trainer with all necessary components.
        
        Args:
            agent (Agent): Learning agent instance
            world (World): Environment instance
            config (Config): Training configuration parameters
        """
        self.agent = agent
        self.world = world
        self.config = config
        self.episode_rewards = []      # list[int]: Cumulative rewards per episode
        self.current_episode = 0       # int: Episode counter
        self.latest_path = None        # tuple[list[float], list[float]]: (x_coords, y_coords)
    
    def run_one_episode(self):
        """
        Executes a single training episode with configurable path crossing.
        
        Updates episode_rewards, current_episode, and latest_path.
        """
        state = self.world.get_start_state()
        
        # Track visited positions if path crossing disabled
        visited = {state.position} if not self.config.allow_path_crossing else set()
        
        total_reward = 0
        # Path coordinates offset by 0.5 for centre of cell
        path_x, path_y = [state.col + 0.5], [state.row + 0.5]
        
        # Execute steps until terminal state or step limit
        for step_num in range(self.config.steps):
            # Determine available actions based on path crossing setting
            if self.config.allow_path_crossing:
                available = state.get_valid_actions()
            else:
                # Filter out actions leading to visited states
                available = [
                    a for a in state.get_valid_actions()
                    if self.world.get_next_state(state, a).position not in visited
                ]
            
            if not available:
                break  # No valid moves available
            
            try:
                # Select and execute action
                action = self.agent.choose_action(state, available)
                next_state = self.world.get_next_state(state, action)
                
                # Learn from transition
                self.agent.update_q_value(state, action, next_state)
                
                # Update visited set if path crossing disabled
                if not self.config.allow_path_crossing:
                    visited.add(next_state.position)
                
                # Move to next state and record path
                state = next_state
                total_reward += state.get_reward()
                path_x.append(state.col + 0.5)
                path_y.append(state.row + 0.5)
                
                # Check for terminal state
                if state.is_terminal():
                    break
                    
            except ValueError:
                break  # No actions available
        
        # Record episode results
        self.episode_rewards.append(total_reward)
        self.latest_path = (path_x, path_y)
        self.current_episode += 1
    
    def train(self):
        """
        Executes complete training loop for configured number of episodes.
        
        Runs remaining episodes until config.episodes is reached.
        """
        episodes_to_run = self.config.episodes - self.current_episode
        for episode in range(episodes_to_run):
            self.run_one_episode()
        print("\nTraining finished.")
