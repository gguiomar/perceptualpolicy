from envs.gridworld import GridWorld
from agents.policygradient import MaxEntAgent
from utils.visualization import plot_metrics, visualize_policy, plot_trajectories, plot_state_visitation_heatmap

if __name__ == "__main__":
    env = GridWorld(grid_size=10, stochastic=False, noise=0)
    agent = MaxEntAgent(
        state_dim=2,
        action_dim=4,
        hidden_dim=200,
        temperature=0.01,
        gamma=0.99
    )
    rewards, losses, entropies = agent.train(env, num_episodes=2000, max_steps=100)
    plot_metrics(rewards, losses, entropies)
    visualize_policy(env, agent)
    plot_trajectories(env, agent, num_trajectories=10, max_steps=100)
    plot_state_visitation_heatmap(env, agent, max_steps=100)
# %%
