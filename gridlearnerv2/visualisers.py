from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

from config import Config
from simulation_core import Action, Agent, World, Trainer

class Visualiser:
    def __init__(self, agent: Agent, world: World, config: Config, episode_rewards: list):
        self.agent, self.world, self.config = agent, world, config
        self.episode_rewards = episode_rewards

    def plot_q_values(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Q-Values per State-Action', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

        for r in range(self.world.size):
            for c in range(self.world.size):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='lightgray', lw=1))
                pos = (r, c)
                if pos in self.world.walls:
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, facecolor='#6c757d', edgecolor='black'))
                    ax.text(c + 0.5, r + 0.5, '■', ha='center', va='center', fontsize=20, color='black')
                    continue
                if pos == self.config.start_pos:
                    ax.text(c + 0.1, r + 0.1, 'S', ha='left', va='top', fontsize=10, color='#007bff', fontweight='bold')
                state = self.world.states[pos]
                if state.get_reward() == 1:
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20, color='#28a745', fontweight='bold')
                    continue
                if state.get_reward() == -1:
                    ax.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', fontsize=20, color='#dc3545', fontweight='bold')
                    continue

                q_vals = state.q_values
                bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.8)
                positions = {
                    'Up':    {'x': c + 0.5, 'y': r + 0.2, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.5, 'arrow_y': r + 0.4, 'dx': 0, 'dy': -0.15},
                    'Right': {'x': c + 0.8, 'y': r + 0.5, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.6, 'arrow_y': r + 0.5, 'dx': 0.15, 'dy': 0},
                    'Down':  {'x': c + 0.5, 'y': r + 0.8, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.5, 'arrow_y': r + 0.6, 'dx': 0, 'dy': 0.15},
                    'Left':  {'x': c + 0.2, 'y': r + 0.5, 'ha': 'center', 'va': 'center', 'arrow_x': c + 0.4, 'arrow_y': r + 0.5, 'dx': -0.15, 'dy': 0}
                }
                for i in range(4):
                    action, pos = Action(i), positions[Action(i).name()]
                    ax.text(pos['x'], pos['y'], f"{q_vals[i]:.2f}", ha=pos['ha'], va=pos['va'], fontsize=9, color='black', bbox=bbox_props)
                    ax.arrow(pos['arrow_x'], pos['arrow_y'], pos['dx'], pos['dy'], head_width=0.1, color='#007bff', alpha=0.7)

        ax.set_xlim(0, self.world.size)
        ax.set_ylim(self.world.size, 0)
        plt.tight_layout()
        plt.show()

    def display_results(self):
        print("\nLearned Policy (Best move from each cell):")
        for r in range(self.world.size):
            row_str = ""
            for c in range(self.config.grid_size):
                pos = (r, c)
                if pos in self.world.walls: row_str += " ■  "
                elif pos in self.world.states:
                    state = self.world.states[pos]
                    if state.get_reward() == 1: row_str += " G  "
                    elif state.get_reward() == -1: row_str += " X  "
                    else: row_str += f" {state.get_max_action().symbol()}  "
            print(row_str)

        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards[-100:])
            success_rate = sum(r > 0 for r in self.episode_rewards) / len(self.episode_rewards) * 100
            print(f"\nFinal {len(self.episode_rewards)} episode average reward: {final_avg_reward:.2f}")
            print(f"Success rate (reaching goal): {success_rate:.1f}%")
        self.plot_q_values()


class LiveVisualiser:
    def __init__(self, trainer: Trainer, world: World, config: Config):
        self.trainer, self.world, self.config = trainer, world, config
        self.fig, self.ax_grid = plt.subplots(1, 1, figsize=(10, 8))
        self.frame_info, self.path_line = [], None
        self.last_success_rate, self.anim, self.is_paused, self.pause_button = 0.0, None, False, None
        self.arrow_base_size, self.arrow_size_multiplier, self.arrow_head_scale = 0.05, 0.1, 0.8
        self.transparency_base, self.transparency_multiplier, self.q_value_fontsize = 0.3, 0.7, 7
        self.animation_interval_ms = 100
        self.arrow_params = {
            0: {'dx': 0, 'dy': -0.15, 'x_offset': 0.5, 'y_offset': 0.35},
            1: {'dx': 0.15, 'dy': 0, 'x_offset': 0.65, 'y_offset': 0.5},
            2: {'dx': 0, 'dy': 0.15, 'x_offset': 0.5, 'y_offset': 0.65},
            3: {'dx': -0.15, 'dy': 0, 'x_offset': 0.35, 'y_offset': 0.5}}
        self.text_positions = {}
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                for i in range(4):
                    params = self.arrow_params[i]
                    text_x = c + params['x_offset'] + params['dx'] * 0.7
                    text_y = r + params['y_offset'] + params['dy'] * 0.7
                    self.text_positions[(r, c, i)] = (text_x, text_y)
        self.setup_plot()

    def setup_plot(self):
        self.fig.subplots_adjust(bottom=0.15, top=0.9)
        self.ax_grid.set_title('Q-Learning Progress', fontsize=14, pad=20)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.set_xticks([]); self.ax_grid.set_yticks([])
        self.ax_grid.invert_yaxis()
        self.ax_grid.set_xlim(0, self.world.size); self.ax_grid.set_ylim(self.world.size, 0)
        for r in range(self.world.size):
            for c in range(self.world.size):
                self.ax_grid.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='black', lw=0.5))
                pos = (r, c)
                if pos in self.world.walls:
                    self.ax_grid.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, facecolor='gray', edgecolor='black', lw=0.5))
                    self.ax_grid.text(c + 0.5, r + 0.5, '■', ha='center', va='center', fontsize=20, color='darkgray', weight='bold')
                elif pos == self.config.start_pos:
                    self.ax_grid.text(c + 0.1, r + 0.1, 'S', ha='left', va='top', fontsize=10, color='blue', weight='bold')
                elif pos in self.world.states:
                    state = self.world.states[pos]
                    if state.get_reward() == 1: self.ax_grid.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontsize=20, color='green', weight='bold')
                    elif state.get_reward() == -1: self.ax_grid.text(c + 0.5, r + 0.5, 'X', ha='center', va='center', fontsize=20, color='red', weight='bold')
        
        self.episode_text_artist = self.fig.text(0.02, 0.95, '', fontsize=12, fontweight='bold')
        self.success_text_artist = self.fig.text(0.98, 0.95, '', ha='right', fontsize=12, fontweight='bold')

    def toggle_pause(self, event):
        if self.is_paused:
            self.anim.resume()
            self.pause_button.label.set_text('Pause')
        else:
            self.anim.pause()
            self.pause_button.label.set_text('Resume')
        self.is_paused = not self.is_paused

    def update_frame(self, frame: int) -> list:
        if self.trainer.current_episode < self.config.episodes: self.trainer.run_one_episode()
        for info in self.frame_info: info.remove()
        self.frame_info.clear()
        if self.path_line: self.path_line.remove(); self.path_line = None
        if self.trainer.latest_path:
            path_x, path_y = self.trainer.latest_path
            if len(path_x) > 1:
                self.path_line, = self.ax_grid.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
        
        all_q_values = [q for s in self.world.states.values() if not s.is_terminal() for q in s.q_values]
        max_abs_q = max(abs(q) for q in all_q_values) if all_q_values else 1.0
        if max_abs_q == 0: max_abs_q = 1.0
        text_bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8)

        for r in range(self.world.size):
            for c in range(self.world.size):
                if (r, c) in self.world.walls: continue
                state = self.world.states[(r, c)]
                if state.is_terminal(): continue
                for i in range(4):
                    q_val = state.q_values[i]
                    normalized_q = (q_val / max_abs_q) * 0.5 + 0.5
                    size_factor = abs(normalized_q - 0.5) * 2
                    arrow_size = self.arrow_base_size + size_factor * self.arrow_size_multiplier
                    color = plt.cm.RdYlGn(normalized_q)
                    transparency = self.transparency_base + size_factor * self.transparency_multiplier
                    params = self.arrow_params[i]
                    arrow = self.ax_grid.arrow(c + params['x_offset'], r + params['y_offset'], params['dx'] * size_factor, params['dy'] * size_factor, head_width=arrow_size, head_length=arrow_size * self.arrow_head_scale, fc=color, ec=color, alpha=transparency)
                    self.frame_info.append(arrow)
                    text_x, text_y = self.text_positions[(r, c, i)]
                    text = self.ax_grid.text(text_x, text_y, f"{q_val:.1f}", ha='center', va='center', fontsize=self.q_value_fontsize, bbox=text_bbox)
                    self.frame_info.append(text)

        self.episode_text_artist.set_text(f"Episode: {self.trainer.current_episode}/{self.config.episodes}")
        rewards = self.trainer.episode_rewards
        if rewards and (self.trainer.current_episode % 10 == 0 or self.trainer.current_episode == self.config.episodes):
            self.last_success_rate = sum(r > 0 for r in rewards) / len(rewards) * 100
        self.success_text_artist.set_text(f"Success Rate: {self.last_success_rate:.1f}%")
        
        return self.frame_info + [self.path_line, self.episode_text_artist, self.success_text_artist]

    def animate_training(self):
        self.anim = FuncAnimation(self.fig, self.update_frame, frames=self.config.episodes + 1, interval=self.animation_interval_ms, repeat=False, blit=False)
        button_ax = self.fig.add_axes([0.45, 0.05, 0.1, 0.05])
        self.pause_button = Button(button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)
        return self.anim
