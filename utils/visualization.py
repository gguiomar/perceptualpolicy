import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random


def plot_metrics(rewards, losses, entropies):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(losses)
    axs[0].set_title("Policy Loss")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].plot(entropies)
    axs[1].set_title("Policy Entropy")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Entropy")
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(rewards)
    axs[2].set_title("Rewards")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/metrics.png')


def plot_mean_metrics(rewards, losses, entropies, window=100):
    def moving_average(data, window):
        return [np.mean(data[i:i+window]) for i in range(len(data) - window)]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(moving_average(losses, window))
    axs[0].set_title("Policy Loss (Moving Average)")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].plot(moving_average(entropies, window))
    axs[1].set_title("Policy Entropy (Moving Average)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Entropy")
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(moving_average(rewards, window))
    axs[2].set_title("Rewards (Moving Average)")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/mean_metrics.png')

def visualize_policy(env, agent):
    action_maps = {
        'UP': np.zeros((env.grid_size, env.grid_size)),
        'DOWN': np.zeros((env.grid_size, env.grid_size)),
        'LEFT': np.zeros((env.grid_size, env.grid_size)),
        'RIGHT': np.zeros((env.grid_size, env.grid_size))
    }
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            state = np.array([i, j], dtype=np.float32) / (env.grid_size - 1)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                logits = agent.policy_net(state_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            action_maps['UP'][j, i] = probs[env.UP]
            action_maps['DOWN'][j, i] = probs[env.DOWN]
            action_maps['LEFT'][j, i] = probs[env.LEFT]
            action_maps['RIGHT'][j, i] = probs[env.RIGHT]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for ax, (action_name, action_map) in zip(axes, action_maps.items()):
        im = ax.imshow(action_map, origin='lower', cmap='viridis')
        ax.set_title(f"Action: {action_name}")
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/policy_visualization.png')

def plot_trajectories(env, agent, num_trajectories=10, max_steps=100):
    plt.figure(figsize=(8, 8))
    
    for i in range(env.grid_size + 1):
        plt.axhline(i, color='black', linewidth=0.5, alpha=0.5)
        plt.axvline(i, color='black', linewidth=0.5, alpha=0.5)
    
    for obs in env.obstacles:
        plt.gca().add_patch(plt.Rectangle((obs[0], obs[1]), 1, 1, color='gray', alpha=0.5))
    
    plt.scatter(env.start[0] + 0.5, env.start[1] + 0.5, color='blue', s=100, marker='o', label='Start')
    plt.scatter(env.goal[0] + 0.5, env.goal[1] + 0.5, color='green', s=100, marker='*', label='Goal')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    for i in range(num_trajectories):
        state = env.reset()
        positions = [env.agent_pos.copy()]
        
        for step in range(max_steps):
            action, _, _ = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            positions.append(env.agent_pos.copy())
            state = next_state
            
            if done:
                break
        

        positions = np.array(positions)
        

        plt.plot(positions[:, 0] + 0.5, positions[:, 1] + 0.5, 'o-', 
                 color=colors[i], markersize=4, alpha=0.7,
                 label=f'Trajectory {i+1}')
    
    plt.xlim(0, env.grid_size)
    plt.ylim(0, env.grid_size)
    plt.title('Sample Trajectories with Trained Policy')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:5], labels[:5], loc='upper left', fontsize='small')
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/trajectories.png')

def plot_state_visitation_heatmap(env, agent, num_episodes=100, max_steps=100):
    """
    Plots a heatmap showing how frequently each state is visited by the learned policy.
    
    Args:
        env: The environment
        agent: The trained agent
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    # Initialize visitation count matrix
    visitation_counts = np.zeros((env.grid_size, env.grid_size))
    
    for episode in range(num_episodes):
        state = env.reset()
        
        # Record initial position
        visitation_counts[env.agent_pos[1], env.agent_pos[0]] += 1
        
        for step in range(max_steps):
            action, _, _ = agent.select_action(state)
            next_state, _, done, _ = env.step(action)
            
            # Record position
            visitation_counts[env.agent_pos[1], env.agent_pos[0]] += 1
            
            state = next_state
            
            if done:
                break
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    plt.imshow(visitation_counts, cmap='viridis', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Visit Count')
    
    # Draw grid lines
    for i in range(env.grid_size + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        plt.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Mark start and goal
    plt.scatter(env.start[0], env.start[1], color='blue', s=100, marker='o', label='Start')
    plt.scatter(env.goal[0], env.goal[1], color='green', s=100, marker='*', label='Goal')
    
    # Mark obstacles
    for obs in env.obstacles:
        plt.gca().add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                         fill=True, color='gray', alpha=0.7))
    
    # Set ticks and labels
    plt.xticks(np.arange(env.grid_size))
    plt.yticks(np.arange(env.grid_size))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('State Visitation Frequency Heatmap')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/visitation_heatmap.png')



    import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)


def compare_training_curves(agent_metrics, window=50, save_path=None):
    """
    Compare training curves for multiple agents
    
    Args:
        agent_metrics: Dictionary mapping agent names to (rewards, losses, metrics) tuples
        window: Window size for moving average
        save_path: Path to save the plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot rewards
    for agent_name, (rewards, losses, metrics) in agent_metrics.items():
        if len(rewards) >= window:
            smoothed_rewards = moving_average(rewards, window)
            axs[0].plot(smoothed_rewards, label=agent_name)
    
    axs[0].set_title("Reward Comparison (Moving Average)")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend()
    
    # Plot losses
    for agent_name, (rewards, losses, metrics) in agent_metrics.items():
        if len(losses) >= window:
            smoothed_losses = moving_average(losses, window)
            axs[1].plot(smoothed_losses, label=agent_name)
    
    axs[1].set_title("Loss Comparison (Moving Average)")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend()
    
    # Plot metrics (entropy/KL)
    for agent_name, (rewards, losses, metrics) in agent_metrics.items():
        if len(metrics) >= window:
            smoothed_metrics = moving_average(metrics, window)
            metric_label = "Entropy" if "MaxEnt" in agent_name else "KL Divergence"
            axs[2].plot(smoothed_metrics, label=f"{agent_name} ({metric_label})")
    
    axs[2].set_title("Additional Metrics (Moving Average)")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Metric Value")
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def compare_policy_visualizations(env, agent_dict, save_path=None):
    """
    Compare policy heatmaps for multiple agents
    
    Args:
        env: The environment
        agent_dict: Dictionary mapping agent names to agent instances
        save_path: Path to save the plot
    """
    num_agents = len(agent_dict)
    fig, axes = plt.subplots(num_agents, 4, figsize=(16, 4 * num_agents))
    
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    
    for agent_idx, (agent_name, agent) in enumerate(agent_dict.items()):
        action_maps = {
            'UP': np.zeros((env.grid_size, env.grid_size)),
            'RIGHT': np.zeros((env.grid_size, env.grid_size)),
            'DOWN': np.zeros((env.grid_size, env.grid_size)),
            'LEFT': np.zeros((env.grid_size, env.grid_size))
        }
        
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                # Create normalized state representation
                state = np.array([i, j], dtype=np.float32) / (env.grid_size - 1)
                state_tensor = torch.FloatTensor(state).to(agent.device)
                
                with torch.no_grad():
                    if agent.policy_type == "rnn":
                        # For RNN, we need a proper sequence and hidden state
                        logits, _ = agent.policy_net(state_tensor)
                    elif agent.policy_type == "transformer":
                        # For transformer, format state correctly
                        logits = agent.policy_net(state_tensor)
                    else:
                        # Standard MLP forward pass
                        logits = agent.policy_net(state_tensor)
                    
                    probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                for a, action_name in enumerate(action_names):
                    action_maps[action_name][j, i] = probs[a]
        
        # Get agent-specific axes
        if num_agents == 1:
            agent_axes = axes
        else:
            agent_axes = axes[agent_idx]
        
        # Plot each action's probability
        for ax_idx, (action_name, action_map) in enumerate(zip(action_names, action_maps.values())):
            ax = agent_axes[ax_idx] if num_agents > 1 else axes[ax_idx]
            im = ax.imshow(action_map, origin='lower', cmap='viridis', vmin=0, vmax=1)
            
            if agent_idx == 0:
                ax.set_title(f"Action: {action_name}")
            
            # Add grid lines
            for k in range(env.grid_size):
                ax.axhline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)
                ax.axvline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)
            
            # Mark start, goal and obstacles
            for obs in env.obstacles:
                ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                           fill=True, color='gray', alpha=0.7))
            
            ax.scatter(env.start[0], env.start[1], color='blue', s=80, marker='o')
            ax.scatter(env.goal[0], env.goal[1], color='green', s=80, marker='*')
            
            # Set axis labels
            if agent_idx == num_agents - 1:
                ax.set_xlabel('X')
            if ax_idx == 0:
                ax.set_ylabel(f'{agent_name}\nY')
            
            # Add colorbar
            if ax_idx == 3:
                fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def compare_trajectories(env, agent_dict, num_trajectories=5, max_steps=100, save_path=None):
    """
    Compare trajectories for multiple agents
    
    Args:
        env: The environment
        agent_dict: Dictionary mapping agent names to agent instances
        num_trajectories: Number of trajectories per agent
        max_steps: Maximum steps per trajectory
        save_path: Path to save the plot
    """
    num_agents = len(agent_dict)
    fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 6))
    
    # If only one agent, make axes iterable
    if num_agents == 1:
        axes = [axes]
    
    # Generate color palette for trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    results = {}
    
    for agent_idx, (agent_name, agent) in enumerate(agent_dict.items()):
        ax = axes[agent_idx]
        
        # Draw grid
        for i in range(env.grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Draw obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                       fill=True, color='gray', alpha=0.7))
        
        # Mark start and goal
        ax.scatter(env.start[0], env.start[1], color='blue', s=100, marker='o', label='Start')
        ax.scatter(env.goal[0], env.goal[1], color='green', s=100, marker='*', label='Goal')
        
        # Run and plot trajectories
        trajectory_lengths = []
        success_count = 0
        
        for i in range(num_trajectories):
            state = env.reset()
            positions = [env.agent_pos.copy()]
            
            hidden_state = None
            if agent.policy_type == "rnn":
                hidden_state = agent.policy_net.init_hidden()
            
            for step in range(max_steps):
                if agent.policy_type == "rnn":
                    action, _, _, h = agent.select_action(state, hidden_state)
                    hidden_state = h
                else:
                    action, _, _ = agent.select_action(state)
                    
                next_state, reward, done, _ = env.step(action)
                positions.append(env.agent_pos.copy())
                state = next_state
                
                if done:
                    success_count += 1
                    break
            
            trajectory_lengths.append(len(positions))
            positions = np.array(positions)
            
            ax.plot(positions[:, 0], positions[:, 1], 'o-', 
                   color=colors[i], markersize=5, alpha=0.7)
        
        # Set plot limits and labels
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_title(f'{agent_name}\nSuccess: {success_count}/{num_trajectories}\nAvg Length: {np.mean(trajectory_lengths):.1f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        ax.set_aspect('equal')
        
        # Store results
        results[agent_name] = (success_count / num_trajectories, np.mean(trajectory_lengths))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    return results


def compare_visitation_heatmaps(env, agent_dict, num_episodes=50, max_steps=100, save_path=None):
    """
    Compare state visitation heatmaps for multiple agents
    
    Args:
        env: The environment
        agent_dict: Dictionary mapping agent names to agent instances
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        save_path: Path to save the plot
    """
    num_agents = len(agent_dict)
    fig, axes = plt.subplots(1, num_agents, figsize=(5 * num_agents, 5))
    
    # If only one agent, make axes iterable
    if num_agents == 1:
        axes = [axes]
    
    for agent_idx, (agent_name, agent) in enumerate(agent_dict.items()):
        ax = axes[agent_idx]
        
        # Initialize visitation count matrix
        visitation_counts = np.zeros((env.grid_size, env.grid_size))
        
        # Run episodes and count state visitations
        for episode in range(num_episodes):
            state = env.reset()
            
            # Record initial position
            visitation_counts[env.agent_pos[1], env.agent_pos[0]] += 1
            
            hidden_state = None
            if agent.policy_type == "rnn":
                hidden_state = agent.policy_net.init_hidden()
            
            for step in range(max_steps):
                if agent.policy_type == "rnn":
                    action, _, _, h = agent.select_action(state, hidden_state)
                    hidden_state = h
                else:
                    action, _, _ = agent.select_action(state)
                    
                next_state, _, done, _ = env.step(action)
                
                # Record position
                visitation_counts[env.agent_pos[1], env.agent_pos[0]] += 1
                
                state = next_state
                
                if done:
                    break
        
        # Plot heatmap with logarithmic scale for better visualization
        im = ax.imshow(np.log1p(visitation_counts), cmap='viridis', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log(Visit Count + 1)')
        
        # Draw grid lines
        for i in range(env.grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Mark start and goal
        ax.scatter(env.start[0], env.start[1], color='blue', s=100, marker='o', label='Start')
        ax.scatter(env.goal[0], env.goal[1], color='green', s=100, marker='*', label='Goal')
        
        # Mark obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                       fill=True, color='gray', alpha=0.7))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(env.grid_size))
        ax.set_yticks(np.arange(env.grid_size))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'State Visitation: {agent_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def plot_single_training_curve(rewards, losses, metrics, agent_name="Agent", window=50, save_path=None):
    """
    Plot training curves for a single agent
    
    Args:
        rewards: List of rewards per episode
        losses: List of losses per update
        metrics: List of additional metrics per update
        agent_name: Name of the agent
        window: Window size for moving average
        save_path: Path to save the plot
    """
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Ensure we have enough data for the moving average
    if len(losses) >= window:
        # Plot loss
        smoothed_losses = moving_average(losses, window)
        axs[0].plot(smoothed_losses)
        axs[0].set_title(f"{agent_name} Policy Loss")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Loss")
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot metric (entropy or KL)
        metric_name = "Entropy" if "MaxEnt" in agent_name else "KL Divergence"
        smoothed_metrics = moving_average(metrics, window)
        axs[1].plot(smoothed_metrics)
        axs[1].set_title(f"{agent_name} {metric_name}")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel(metric_name)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Plot rewards
        smoothed_rewards = moving_average(rewards, window)
        axs[2].plot(smoothed_rewards)
        axs[2].set_title(f"{agent_name} Rewards")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Reward")
        axs[2].grid(True, linestyle='--', alpha=0.7)
    else:
        axs[0].text(0.5, 0.5, f"Not enough data for {window}-episode window", 
                   horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()


def visualize_single_policy(env, agent, save_path=None):
    """
    Visualize the policy of a single agent as action probability heatmaps
    
    Args:
        env: The environment
        agent: The trained agent
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    action_maps = {
        'UP': np.zeros((env.grid_size, env.grid_size)),
        'RIGHT': np.zeros((env.grid_size, env.grid_size)),
        'DOWN': np.zeros((env.grid_size, env.grid_size)),
        'LEFT': np.zeros((env.grid_size, env.grid_size))
    }
    
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            # Create normalized state representation
            state = np.array([i, j], dtype=np.float32) / (env.grid_size - 1)
            state_tensor = torch.FloatTensor(state).to(agent.device)
            
            with torch.no_grad():
                if agent.policy_type == "rnn":
                    # For RNN, handle properly
                    logits, _ = agent.policy_net(state_tensor)
                else:
                    # For MLP and Transformer
                    logits = agent.policy_net(state_tensor)
                
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
            for a, action_name in enumerate(action_names):
                action_maps[action_name][j, i] = probs[a]
    
    # Plot each action's probability
    for ax, (action_name, action_map) in zip(axes, action_maps.items()):
        im = ax.imshow(action_map, origin='lower', cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Action: {action_name}")
        
        # Add grid lines
        for k in range(env.grid_size):
            ax.axhline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Mark start, goal and obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                       fill=True, color='gray', alpha=0.7))
        
        ax.scatter(env.start[0], env.start[1], color='blue', s=100, marker='o')
        ax.scatter(env.goal[0], env.goal[1], color='green', s=100, marker='*')
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        fig.colorbar(im, ax=ax)
    
    plt.suptitle(f"Policy Visualization: {agent.__class__.__name__} with {agent.policy_type} policy")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()