import torch
from envs.gridworld import GridWorld
from agents.maxent_agent import MaxEntAgent
from utils.visualization import plot_metrics, plot_mean_metrics, visualize_policy, plot_trajectories, plot_state_visitation_heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = GridWorld(grid_size=10, stochastic=False, noise=0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = MaxEntAgent("mlp", state_dim, action_dim, hidden_dim=64, lr=0.01, temperature=0.1, gamma=0.99)
rewards, losses, entropies = agent.train(env, num_episodes=1000, max_steps=100)
plot_mean_metrics(rewards, losses, entropies)
visualize_policy(env, agent)
plot_trajectories(env, agent, num_trajectories=10, max_steps=100)
plot_state_visitation_heatmap(env, agent, max_steps=100)