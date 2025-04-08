# utils/avoidance_visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

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
    ax.set_xticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
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
    visitation_counts = np.zeros((env.grid_size, env.grid_size))
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
            if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
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
    ax.set_xticks(np.arange(env.grid_size) + 0.5)
    ax.set_yticks(np.arange(env.grid_size) + 0.5)
    ax.set_xticklabels(np.arange(env.grid_size))
    ax.set_yticklabels(np.arange(env.grid_size))
    plt.setp(ax.get_yticklabels(), rotation=0)


    if save_path:
        plt.savefig(save_path)
        print(f"Avoidance heatmap saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
