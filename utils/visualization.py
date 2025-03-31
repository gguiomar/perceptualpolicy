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