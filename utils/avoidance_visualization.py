# utils/avoidance_visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation

def plot_avoidance_training_curves(rewards, losses, metrics, metric_name,
                                   avoidance_rates, shock_rates,
                                   smooth_window=50, save_path=None):
    """
    Plots smoothed training curves specifically for the avoidance task,
    including avoidance and shock rates.

    Args:
        rewards: List of rewards
        losses: List of losses
        metrics: List of the third metric (e.g., entropy, KL)
        metric_name: Name of the third metric
        avoidance_rates: List of booleans/ints (1 for avoided, 0 otherwise).
        shock_rates: List of booleans/ints (1 for shocked, 0 otherwise).
        smooth_window: Window size for moving average smoothing.
        save_path: Path to save the plot. If None, displays the plot.
    """
    num_plots = 5
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15), sharex=True)
    x_range = range(len(rewards))

    # --- Smoothing function ---
    def smooth(data, window):
        if len(data) < window:
            return data, range(len(data))
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        x_axis = range(window - 1, len(data))
        return smoothed, x_axis

    # --- Plot Rewards ---
    smooth_rewards, reward_x = smooth(rewards, smooth_window)
    axs[0].plot(x_range, rewards, label='Raw Reward', alpha=0.3)
    axs[0].plot(reward_x, smooth_rewards, label=f'Smoothed Reward (w={smooth_window})')
    axs[0].set_ylabel('Episode Reward')
    axs[0].set_title('Training Rewards')
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot Losses ---
    smooth_losses, loss_x = smooth(losses, smooth_window)
    axs[1].plot(x_range, losses, label='Raw Loss', alpha=0.3)
    axs[1].plot(loss_x, smooth_losses, label=f'Smoothed Loss (w={smooth_window})')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training Loss')
    axs[1].legend()
    axs[1].grid(True)

    # --- Plot Custom Metric ---
    smooth_metrics, metric_x = smooth(metrics, smooth_window)
    axs[2].plot(x_range, metrics, label=f'Raw {metric_name}', alpha=0.3)
    axs[2].plot(metric_x, smooth_metrics, label=f'Smoothed {metric_name} (w={smooth_window})')
    axs[2].set_ylabel(metric_name)
    axs[2].set_title(f'Training {metric_name}')
    axs[2].legend()
    axs[2].grid(True)

    # --- Plot Avoidance Rate ---
    smooth_avoid, avoid_x = smooth(np.array(avoidance_rates) * 100, smooth_window)
    axs[3].plot(x_range, np.array(avoidance_rates) * 100, label='Raw Avoid %', alpha=0.3)
    axs[3].plot(avoid_x, smooth_avoid, label=f'Smoothed Avoid % (w={smooth_window})')
    axs[3].set_ylabel('Avoidance Rate (%)')
    axs[3].set_title('Avoidance Performance')
    axs[3].set_ylim(-5, 105) # Set y-axis limits for percentage
    axs[3].legend()
    axs[3].grid(True)

    # --- Plot Shock Rate ---
    smooth_shock, shock_x = smooth(np.array(shock_rates) * 100, smooth_window)
    axs[4].plot(x_range, np.array(shock_rates) * 100, label='Raw Shock %', alpha=0.3)
    axs[4].plot(shock_x, smooth_shock, label=f'Smoothed Shock % (w={smooth_window})')
    axs[4].set_ylabel('Shock Rate (%)')
    axs[4].set_xlabel('Episode')
    axs[4].set_title('Shock Performance')
    axs[4].set_ylim(-5, 105) # Set y-axis limits for percentage
    axs[4].legend()
    axs[4].grid(True)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Avoidance training curves saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_avoidance_trajectories_2d(env, agent, num_trajectories=5, max_steps=100, save_path=None):
    """
    Plots sample trajectories for the ActiveAvoidanceEnv2D.

    Args:
        env: An instance of ActiveAvoidanceEnv2D.
        agent: The trained agent.
        num_trajectories: Number of trajectories to plot.
        max_steps: Maximum steps per trajectory.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    device = agent.device

    task_id = env.current_task_id 

    for i in range(num_trajectories):
        state = env.reset()
        trajectory_x = []
        trajectory_y = []
        trajectory_x.append(env.agent_pos[0])
        trajectory_y.append(env.agent_pos[1])
        done = False
        steps = 0
        ep_info = {}

        # Prepare for RNN/Transformer if needed
        hidden_state = None
        if agent.policy_type in ["rnn", "transformer"] and hasattr(agent.policy_net, 'init_hidden'):
             hidden_state = agent.policy_net.init_hidden()

        while not done and steps < max_steps:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                 if agent.policy_type in ["rnn", "transformer"]:
                     action, _, _, hidden_state = agent.select_action(state, hidden_state)
                 else:
                     action, _, _ = agent.select_action(state)

            state, reward, done, info = env.step(action)
            trajectory_x.append(env.agent_pos[0])
            trajectory_y.append(env.agent_pos[1])
            steps += 1
            if done:
                ep_info = info

        color = 'green' if ep_info.get('avoided', False) else ('red' if ep_info.get('shocked', False) else 'orange')
        label = f"Traj {i+1} ({'Avoided' if ep_info.get('avoided', False) else ('Shocked' if ep_info.get('shocked', False) else 'Timeout')})"
        ax.plot(trajectory_x, trajectory_y, marker='.', linestyle='-', label=label, color=color, alpha=0.7)
        ax.plot(trajectory_x[0], trajectory_y[0], marker='o', color=color, markersize=8) # Start point

    # Draw center lines and grid
    ax.axvline(env.center_x, color='gray', linestyle='--', lw=1)
    ax.axhline(env.center_y, color='gray', linestyle='--', lw=1)
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    task_str = f"Task {'X' if task_id == 1 else 'Y'}-Shuttle"
    ax.set_title(f"Sample Trajectories ({agent.__class__.__name__} - {agent.policy_type}) - {task_str}")
    ax.legend(fontsize='small')

    if save_path:
        plt.savefig(save_path)
        print(f"Avoidance trajectories plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_avoidance_heatmap_2d(env, agent, num_episodes=100, max_steps=100, save_path=None):
    """
    Plots a heatmap of state visitation frequency for ActiveAvoidanceEnv2D.

    Args:
        env: An instance of ActiveAvoidanceEnv2D.
        agent: The trained agent.
        num_episodes: Number of episodes to simulate for visitation count.
        max_steps: Maximum steps per episode.
        save_path: Path to save the plot.
    """
    visitation_counts = np.zeros((env.height, env.width))
    device = agent.device
    task_id = env.current_task_id 

    print(f"Generating heatmap data for Task {task_id} ({num_episodes} episodes)...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        # Prepare for RNN/Transformer if needed
        hidden_state = None
        if agent.policy_type in ["rnn", "transformer"] and hasattr(agent.policy_net, 'init_hidden'):
             hidden_state = agent.policy_net.init_hidden()

        while not done and steps < max_steps:
            x, y = env.agent_pos
            # Ensure indices are within bounds before incrementing
            if 0 <= x < env.width and 0 <= y < env.height:
                visitation_counts[int(round(y)), int(round(x))] += 1 # Use rounded int indices

            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                 if agent.policy_type in ["rnn", "transformer"]:
                     action, _, _, hidden_state = agent.select_action(state, hidden_state)
                 else:
                     action, _, _ = agent.select_action(state)

            state, _, done, _ = env.step(action)
            steps += 1
        if episode % max(1, (num_episodes // 10)) == 0:
             print(f"Heatmap episode {episode+1}/{num_episodes}")


    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(visitation_counts, cmap="viridis", linewidths=.5, annot=False, fmt=".0f", ax=ax, cbar=True, square=True, origin='lower')

    # Draw center lines
    ax.axvline(env.center_x + 0.5, color='white', linestyle='--', lw=1) # Offset by 0.5 for heatmap indices
    ax.axhline(env.center_y + 0.5, color='white', linestyle='--', lw=1)

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    task_str = f"Task {'X' if task_id == 1 else 'Y'}-Shuttle"
    ax.set_title(f"State Visitation Heatmap ({agent.__class__.__name__} - {agent.policy_type}) - {task_str}")
    # Ensure correct tick labels for grid coordinates
    ax.set_xticks(np.arange(env.width) + 0.5)
    ax.set_yticks(np.arange(env.height) + 0.5)
    ax.set_xticklabels(np.arange(env.width))
    ax.set_yticklabels(np.arange(env.height))
    plt.setp(ax.get_yticklabels(), rotation=0)


    if save_path:
        plt.savefig(save_path)
        print(f"Avoidance heatmap saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_avoidance_trajectory_step_by_step(env, agent, max_steps=100, save_path=None, 
                                          visualization_type='animation', fps=5, 
                                          gallery_save_dir=None):
    """
    Plots a single trajectory for the ActiveAvoidanceEnv2D with step-by-step visualization.
    
    Args:
        env: An instance of ActiveAvoidanceEnv2D.
        agent: The trained agent.
        max_steps: Maximum steps per trajectory.
        save_path: Path to save the animation or first gallery image.
        visualization_type: Either 'animation' or 'gallery'.
        fps: Frames per second for animation.
        gallery_save_dir: Directory to save gallery images. If None, uses save_path directory.
    """
    device = agent.device
    task_id = env.current_task_id
    
    # Collect trajectory data
    state = env.reset()
    trajectory_x = [env.agent_pos[0]]
    trajectory_y = [env.agent_pos[1]]
    actions = []
    rewards = []
    tones = []  # Track which tone was playing
    done = False
    steps = 0
    ep_info = {}
    
    # Prepare for RNN/Transformer if needed
    hidden_state = None
    if agent.policy_type in ["rnn", "transformer"] and hasattr(agent.policy_net, 'init_hidden'):
        hidden_state = agent.policy_net.init_hidden()
    
    while not done and steps < max_steps:
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            if agent.policy_type in ["rnn", "transformer"]:
                action, _, _, hidden_state = agent.select_action(state, hidden_state)
            else:
                action, _, _ = agent.select_action(state)
        
        actions.append(action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        trajectory_x.append(env.agent_pos[0])
        trajectory_y.append(env.agent_pos[1])
        
        # Track which tone was playing
        current_tone = "None"
        if 'tone1_active' in info and info['tone1_active']:
            current_tone = "Tone1"
        elif 'tone2_active' in info and info['tone2_active']:
            current_tone = "Tone2"
        tones.append(current_tone)
        
        steps += 1
        if done:
            ep_info = info
    
    # Determine trajectory outcome color
    color = 'green' if ep_info.get('avoided', False) else ('red' if ep_info.get('shocked', False) else 'orange')
    outcome = 'Avoided' if ep_info.get('avoided', False) else ('Shocked' if ep_info.get('shocked', False) else 'Timeout')
    
    # Function to set up the grid properly
    def setup_grid(ax):
        # Set axis limits to match the grid dimensions
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        
        # Set major ticks at integer positions
        ax.set_xticks(np.arange(0, env.width + 1))
        ax.set_yticks(np.arange(0, env.height + 1))
        
        # Set minor ticks for grid lines
        ax.set_xticks(np.arange(0, env.width + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, env.height + 1, 1), minor=True)
        
        # Draw grid lines
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Draw center lines
        ax.axvline(env.center_x, color='gray', linestyle='--', lw=1)
        ax.axhline(env.center_y, color='gray', linestyle='--', lw=1)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal', adjustable='box')
        
        # Set labels
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
    
    if visualization_type == 'animation':
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjusted for 20x10 grid
        
        # Set up the grid
        setup_grid(ax)
        
        task_str = f"Task {'Tone1' if task_id == 1 else 'Tone2'}-Shuttle"
        ax.set_title(f"Step-by-Step Trajectory ({agent.__class__.__name__} - {agent.policy_type}) - {task_str}")
        
        # Initialize empty line and point
        line, = ax.plot([], [], marker='.', linestyle='-', color=color, alpha=0.7)
        point, = ax.plot([], [], marker='o', color=color, markersize=8)
        step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        action_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        tone_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)  # Add tone text
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            step_text.set_text('')
            action_text.set_text('')
            reward_text.set_text('')
            tone_text.set_text('')  # Initialize tone text
            return line, point, step_text, action_text, reward_text, tone_text
        
        def update(frame):
            # Update line with trajectory up to current frame
            line.set_data(trajectory_x[:frame+1], trajectory_y[:frame+1])
            
            # Update point to current position - FIX: Use lists for x and y data
            point.set_data([trajectory_x[frame]], [trajectory_y[frame]])
            
            # Update text information
            step_text.set_text(f'Step: {frame}/{len(trajectory_x)-1}')
            if frame < len(actions):
                action_text.set_text(f'Action: {actions[frame]}')
                reward_text.set_text(f'Reward: {rewards[frame]:.2f}')
                tone_text.set_text(f'Tone: {tones[frame]}')  # Update tone text
            else:
                action_text.set_text('')
                reward_text.set_text('')
                tone_text.set_text('')
            
            return line, point, step_text, action_text, reward_text, tone_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(trajectory_x), 
                            init_func=init, blit=True, interval=1000/fps)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Trajectory animation saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    elif visualization_type == 'gallery':
        # Create gallery of images
        if gallery_save_dir is None and save_path is not None:
            import os
            gallery_save_dir = os.path.dirname(save_path)
            if not gallery_save_dir:
                gallery_save_dir = '.'
        
        for frame in range(len(trajectory_x)):
            fig, ax = plt.subplots(figsize=(10, 5))  # Adjusted for 20x10 grid
            
            # Set up the grid
            setup_grid(ax)
            
            task_str = f"Task {'X' if task_id == 1 else 'Y'}-Shuttle"
            ax.set_title(f"Step {frame}/{len(trajectory_x)-1} - {outcome}")
            
            # Plot trajectory up to current frame
            ax.plot(trajectory_x[:frame+1], trajectory_y[:frame+1], 
                   marker='.', linestyle='-', color=color, alpha=0.7)
            
            # Plot current position
            ax.plot(trajectory_x[frame], trajectory_y[frame], 
                   marker='o', color=color, markersize=8)
            
            # Add step information
            if frame < len(actions):
                ax.text(0.02, 0.95, f'Action: {actions[frame]}', transform=ax.transAxes)
                ax.text(0.02, 0.90, f'Reward: {rewards[frame]:.2f}', transform=ax.transAxes)
            
            # Save or show the frame
            if gallery_save_dir:
                frame_path = f"{gallery_save_dir}/trajectory_step_{frame:03d}.png"
                plt.savefig(frame_path)
                print(f"Saved frame {frame} to {frame_path}")
                plt.close(fig)
            else:
                plt.show()
                input("Press Enter to continue to next step...")
                plt.close(fig)
    
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")

def plot_multiple_avoidance_trajectories(env, agent, num_runs=4, max_steps=100, save_path=None, 
                                        visualization_type='animation', fps=5, 
                                        gallery_save_dir=None):
    """
    Plots multiple trajectories from different runs side by side.
    
    Args:
        env: An instance of ActiveAvoidanceEnv2D.
        agent: The trained agent.
        num_runs: Number of different runs to visualize.
        max_steps: Maximum steps per trajectory.
        save_path: Path to save the animation or first gallery image.
        visualization_type: Either 'animation' or 'gallery'.
        fps: Frames per second for animation.
        gallery_save_dir: Directory to save gallery images. If None, uses save_path directory.
    """
    device = agent.device
    task_id = env.current_task_id
    
    # Collect data for multiple runs
    all_trajectories = []
    all_actions = []
    all_rewards = []
    all_tones = []
    all_outcomes = []
    all_colors = []
    
    for run in range(num_runs):
        # Collect trajectory data for this run
        state = env.reset()
        trajectory_x = [env.agent_pos[0]]
        trajectory_y = [env.agent_pos[1]]
        actions = []
        rewards = []
        tones = []
        done = False
        steps = 0
        ep_info = {}
        
        # Prepare for RNN/Transformer if needed
        hidden_state = None
        if agent.policy_type in ["rnn", "transformer"] and hasattr(agent.policy_net, 'init_hidden'):
            hidden_state = agent.policy_net.init_hidden()
        
        while not done and steps < max_steps:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                if agent.policy_type in ["rnn", "transformer"]:
                    action, _, _, hidden_state = agent.select_action(state, hidden_state)
                else:
                    action, _, _ = agent.select_action(state)
            
            actions.append(action)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            trajectory_x.append(env.agent_pos[0])
            trajectory_y.append(env.agent_pos[1])
            
            # Track which tone was playing
            current_tone = "None"
            if 'tone1_active' in info and info['tone1_active']:
                current_tone = "Tone1"
            elif 'tone2_active' in info and info['tone2_active']:
                current_tone = "Tone2"
            tones.append(current_tone)
            
            steps += 1
            if done:
                ep_info = info
        
        # Determine trajectory outcome color
        color = 'green' if ep_info.get('avoided', False) else ('red' if ep_info.get('shocked', False) else 'orange')
        outcome = 'Avoided' if ep_info.get('avoided', False) else ('Shocked' if ep_info.get('shocked', False) else 'Timeout')
        
        # Store data for this run
        all_trajectories.append((trajectory_x, trajectory_y))
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_tones.append(tones)
        all_outcomes.append(outcome)
        all_colors.append(color)
    
    # Function to set up the grid properly
    def setup_grid(ax):
        # Set axis limits to match the grid dimensions
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        
        # Set major ticks at integer positions
        ax.set_xticks(np.arange(0, env.width + 1))
        ax.set_yticks(np.arange(0, env.height + 1))
        
        # Set minor ticks for grid lines
        ax.set_xticks(np.arange(0, env.width + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, env.height + 1, 1), minor=True)
        
        # Draw grid lines
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Draw center lines
        ax.axvline(env.center_x, color='gray', linestyle='--', lw=1)
        ax.axhline(env.center_y, color='gray', linestyle='--', lw=1)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal', adjustable='box')
        
        # Set labels
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
    
    if visualization_type == 'animation':
        # Create animation with subplots for each run
        fig, axs = plt.subplots(2, (num_runs + 1) // 2, figsize=(5 * ((num_runs + 1) // 2), 10))
        axs = axs.flatten()
        
        task_str = f"Task {'Tone1' if task_id == 1 else 'Tone2'}-Shuttle"
        fig.suptitle(f"Multiple Trajectories ({agent.__class__.__name__} - {agent.policy_type}) - {task_str}", fontsize=16)
        
        # Initialize empty lines and points for each run
        lines = []
        points = []
        step_texts = []
        action_texts = []
        reward_texts = []
        tone_texts = []
        outcome_texts = []
        
        for i in range(num_runs):
            ax = axs[i]
            setup_grid(ax)
            ax.set_title(f"Run {i+1}")
            
            # Initialize empty line and point
            line, = ax.plot([], [], marker='.', linestyle='-', color=all_colors[i], alpha=0.7)
            point, = ax.plot([], [], marker='o', color=all_colors[i], markersize=8)
            
            # Initialize text elements
            step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=8)
            action_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=8)
            reward_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=8)
            tone_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=8)
            outcome_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                                  color=all_colors[i], fontweight='bold', 
                                  horizontalalignment='center', fontsize=10)
            
            lines.append(line)
            points.append(point)
            step_texts.append(step_text)
            action_texts.append(action_text)
            reward_texts.append(reward_text)
            tone_texts.append(tone_text)
            outcome_texts.append(outcome_text)
        
        # Hide unused subplots if any
        for i in range(num_runs, len(axs)):
            axs[i].set_visible(False)
        
        def init():
            for i in range(num_runs):
                lines[i].set_data([], [])
                points[i].set_data([], [])
                step_texts[i].set_text('')
                action_texts[i].set_text('')
                reward_texts[i].set_text('')
                tone_texts[i].set_text('')
                outcome_texts[i].set_text('')
            return lines + points + step_texts + action_texts + reward_texts + tone_texts + outcome_texts
        
        def update(frame):
            for i in range(num_runs):
                trajectory_x, trajectory_y = all_trajectories[i]
                
                # Update line with trajectory up to current frame
                if frame < len(trajectory_x):
                    lines[i].set_data(trajectory_x[:frame+1], trajectory_y[:frame+1])
                    points[i].set_data([trajectory_x[frame]], [trajectory_y[frame]])
                    
                    # Update text information
                    step_texts[i].set_text(f'Step: {frame}/{len(trajectory_x)-1}')
                    if frame < len(all_actions[i]):
                        action_texts[i].set_text(f'Action: {all_actions[i][frame]}')
                        reward_texts[i].set_text(f'Reward: {all_rewards[i][frame]:.2f}')
                        tone_texts[i].set_text(f'Tone: {all_tones[i][frame]}')
                    
                    # Show outcome at the end
                    if frame == len(trajectory_x) - 1:
                        outcome_texts[i].set_text(all_outcomes[i])
            
            return lines + points + step_texts + action_texts + reward_texts + tone_texts + outcome_texts
        
        # Find the maximum number of steps across all trajectories
        max_frames = max(len(traj[0]) for traj in all_trajectories)
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=max_frames, 
                            init_func=init, blit=True, interval=1000/fps)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Multiple trajectories animation saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    elif visualization_type == 'gallery':
        # Create gallery of images
        if gallery_save_dir is None and save_path is not None:
            import os
            gallery_save_dir = os.path.dirname(save_path)
            if not gallery_save_dir:
                gallery_save_dir = '.'
        
        # Find the maximum number of steps across all trajectories
        max_steps = max(len(traj[0]) for traj in all_trajectories)
        
        for frame in range(max_steps):
            fig, axs = plt.subplots(2, (num_runs + 1) // 2, figsize=(5 * ((num_runs + 1) // 2), 10))
            axs = axs.flatten()
            
            task_str = f"Task {'Tone1' if task_id == 1 else 'Tone2'}-Shuttle"
            fig.suptitle(f"Multiple Trajectories - Step {frame} ({agent.__class__.__name__} - {agent.policy_type}) - {task_str}", fontsize=16)
            
            for i in range(num_runs):
                ax = axs[i]
                setup_grid(ax)
                ax.set_title(f"Run {i+1}")
                
                trajectory_x, trajectory_y = all_trajectories[i]
                
                # Plot trajectory up to current frame
                if frame < len(trajectory_x):
                    ax.plot(trajectory_x[:frame+1], trajectory_y[:frame+1], 
                           marker='.', linestyle='-', color=all_colors[i], alpha=0.7)
                    
                    # Plot current position
                    ax.plot(trajectory_x[frame], trajectory_y[frame], 
                           marker='o', color=all_colors[i], markersize=8)
                    
                    # Add step information
                    if frame < len(all_actions[i]):
                        ax.text(0.02, 0.95, f'Action: {all_actions[i][frame]}', transform=ax.transAxes, fontsize=8)
                        ax.text(0.02, 0.90, f'Reward: {all_rewards[i][frame]:.2f}', transform=ax.transAxes, fontsize=8)
                        ax.text(0.02, 0.85, f'Tone: {all_tones[i][frame]}', transform=ax.transAxes, fontsize=8)
                    
                    # Add outcome text at the end
                    if frame == len(trajectory_x) - 1:
                        ax.text(0.5, 0.95, all_outcomes[i], transform=ax.transAxes, 
                               color=all_colors[i], fontweight='bold', 
                               horizontalalignment='center', fontsize=10)
            
            # Hide unused subplots if any
            for i in range(num_runs, len(axs)):
                axs[i].set_visible(False)
            
            # Save or show the frame
            if gallery_save_dir:
                frame_path = f"{gallery_save_dir}/multiple_trajectories_step_{frame:03d}.png"
                plt.savefig(frame_path)
                print(f"Saved frame {frame} to {frame_path}")
                plt.close(fig)
            else:
                plt.show()
                input("Press Enter to continue to next step...")
                plt.close(fig)
    
    else:
        raise ValueError(f"Unknown visualization type: {visualization_type}")