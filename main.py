import torch
from envs.gridworld import GridWorld
from agents.maxentropy import MaxEntAgent, FisherMaxEntAgent, PPOAgent
from utils.visualization import plot_metrics, plot_mean_metrics, visualize_policy, plot_trajectories, plot_state_visitation_heatmap

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = GridWorld(grid_size=10, stochastic=False, noise=0)
    agent = PPOAgent(
        state_dim=2,
        action_dim=4,
        hidden_dim=64, 
        lr=1e-3, 
        epsilon=0.1, 
        gamma=0.99
    )
    rewards, losses, entropies = agent.train(env, num_episodes=1000, max_steps=100)
    plot_mean_metrics(rewards, losses, entropies)
    visualize_policy(env, agent)
    plot_trajectories(env, agent, num_trajectories=10, max_steps=100)
    plot_state_visitation_heatmap(env, agent, max_steps=100)