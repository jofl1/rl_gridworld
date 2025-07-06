# Q-Learning Configuration Features Implementation Guide

## Overview

This guide outlines how to modify your Q-learning simulation to support configurable behaviours:
1. **Path Crossing Mode**: Toggle between allowing/disallowing revisiting states
2. **Epsilon-Greedy Tie Breaking**: Toggle between strict epsilon-greedy and soft tie-breaking for similar Q-values

## 1. Update Configuration Class

First, extend your `Config` class to include the new behavioural flags:

```python
# In config.py
class Config:
    def __init__(self):
        # Existing configuration...
        self.grid_size = 5
        self.walls = []
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.fire_pos = []
        self.episodes = 1000
        self.steps = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
        # New behavioural configuration
        self.allow_path_crossing = False  # If True, agent can revisit states
        self.use_soft_tiebreaking = True  # If True, randomly choose among similar Q-values
        self.tiebreak_threshold = 0.1     # Percentage threshold for considering Q-values similar
```

## 2. Modify Agent Class

Update the `Agent` class to support soft tie-breaking:

```python
class Agent:
    def __init__(self, world: World, config: Config):
        self.world = world
        self.config = config
    
    def choose_action(self, state: State, action_mask: list[Action] = None) -> Action:
        """
        Enhanced action selection supporting both strict and soft tie-breaking.
        """
        available_actions = action_mask if action_mask is not None else state.get_valid_actions()
        
        if not available_actions:
            raise ValueError(f"No available actions in state {state.to_string()}")
        
        # Standard exploration
        if random.uniform(0, 1) < self.config.exploration_rate:
            return random.choice(available_actions)
        
        # Exploitation with configurable tie-breaking
        q_values = {action: state.get_q_value(action) for action in available_actions}
        max_q = max(q_values.values())
        
        if self.config.use_soft_tiebreaking:
            # Soft tie-breaking: consider actions within threshold of max
            threshold = abs(max_q * self.config.tiebreak_threshold)
            near_optimal_actions = [
                a for a, q in q_values.items() 
                if max_q - q <= threshold
            ]
            return random.choice(near_optimal_actions)
        else:
            # Strict tie-breaking: only exact matches
            optimal_actions = [
                a for a, q in q_values.items() 
                if abs(q - max_q) < 1e-9
            ]
            return random.choice(optimal_actions)
```

## 3. Modify Trainer Class

Update the `Trainer` to support path crossing:

```python
class Trainer:
    def __init__(self, agent: Agent, world: World, config: Config):
        self.agent, self.world, self.config = agent, world, config
        self.episode_rewards = []
        self.current_episode = 0
        self.latest_path = None
    
    def run_one_episode(self):
        """
        Enhanced episode execution with configurable path crossing.
        """
        state = self.world.get_start_state()
        visited = {(state.row, state.col)} if not self.config.allow_path_crossing else set()
        total_reward, done, steps = 0, False, 0
        path_x, path_y = [state.col + 0.5], [state.row + 0.5]
        
        while not done and steps < self.config.steps:
            if self.config.allow_path_crossing:
                # Standard pathfinding: all valid actions available
                available_actions = state.get_valid_actions()
            else:
                # No path crossing: filter out visited states
                available_actions = [
                    action for action in state.get_valid_actions()
                    if self.world.get_next_state(state, action).to_string() not in 
                    {f"({r}, {c})" for r, c in visited}
                ]
            
            if not available_actions:
                break  # Trapped or no valid moves
            
            try:
                action = self.agent.choose_action(state, available_actions)
            except ValueError:
                break
            
            next_state = self.world.get_next_state(state, action)
            self.agent.update_q_value(state, action, next_state)
            
            if not self.config.allow_path_crossing:
                visited.add((next_state.row, next_state.col))
            
            state = next_state
            total_reward += state.get_reward()
            done = state.is_terminal()
            steps += 1
            path_x.append(state.col + 0.5)
            path_y.append(state.row + 0.5)
        
        self.episode_rewards.append(total_reward)
        self.latest_path = (path_x, path_y)
        self.current_episode += 1
```

## 4. GUI Integration (General Guidelines)

### 4.1 Add UI Controls

In your GUI code, add the following controls:

```python
# Pseudo-code for GUI elements
# Add these to your control panel/settings area

# Checkbox for path crossing
path_crossing_checkbox = create_checkbox(
    label="Allow Path Crossing",
    default=config.allow_path_crossing,
    callback=lambda val: setattr(config, 'allow_path_crossing', val)
)

# Checkbox for soft tie-breaking
soft_tiebreak_checkbox = create_checkbox(
    label="Use Soft Tie-Breaking",
    default=config.use_soft_tiebreaking,
    callback=lambda val: setattr(config, 'use_soft_tiebreaking', val)
)

# Slider for tie-break threshold
tiebreak_slider = create_slider(
    label="Tie-Break Threshold (%)",
    min_val=0,
    max_val=50,
    default=config.tiebreak_threshold * 100,
    callback=lambda val: setattr(config, 'tiebreak_threshold', val / 100)
)
```

### 4.2 Update Display

Consider adding visual indicators for the current mode:

```python
# In your rendering/display code
def update_status_display():
    status_text = f"Mode: {'Path Crossing' if config.allow_path_crossing else 'No Revisiting'}"
    status_text += f" | Tie-Breaking: {'Soft' if config.use_soft_tiebreaking else 'Strict'}"
    if config.use_soft_tiebreaking:
        status_text += f" ({config.tiebreak_threshold*100:.1f}%)"
    # Update your status label/display with this text
```

### 4.3 Reset Functionality

When toggling modes, you may want to reset the learning:

```python
def on_mode_change():
    # Reset Q-values when switching modes
    for state in world.states.values():
        state.q_values = np.zeros(4)
    
    # Reset trainer statistics
    trainer.episode_rewards = []
    trainer.current_episode = 0
    
    # Optionally trigger a redraw/update of the visualisation
    update_visualisation()
```

## 5. Usage Examples

### Example 1: Comparing Path Crossing Modes

```python
# Configuration for standard pathfinding
config.allow_path_crossing = True
config.use_soft_tiebreaking = False
# Train and observe behaviour

# Configuration for no-revisiting constraint
config.allow_path_crossing = False
config.use_soft_tiebreaking = False
# Train and observe more exploratory behaviour
```

### Example 2: Exploring Tie-Breaking Effects

```python
# Strict tie-breaking (original behaviour)
config.use_soft_tiebreaking = False
# Observe deterministic exploitation

# Soft tie-breaking with 10% threshold
config.use_soft_tiebreaking = True
config.tiebreak_threshold = 0.1
# Observe more diverse action selection
```
