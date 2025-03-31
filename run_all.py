import torch
import numpy as np
import os
import argparse
from envs.gridworld import GridWorld
from agents.policy_networks import create_policy_network
from agents.base_agent import BaseAgent
from agents.maxent_agent import MaxEntAgent, FisherMaxEntAgent
from agents.ppo_agent import PPOAgent
from agents.trpo_agent import TRPOAgent
from utils.visualization import compare_training_curves, compare_policy_visualizations, compare_trajectories, compare_visitation_heatmaps

# Ensure directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

def train_agent(agent_class, agent_name, policy_type, env, config):
    """
    Train an agent and return the results
    
    Args:
        agent_class: Class of the agent to train
        agent_name: Name of the agent for logging
        policy_type: Type of policy network
        env: Environment to train on
        config: Training configuration
    
    Returns:
        agent: Trained agent
        metrics: (rewards, losses, additional_metrics)
    """
    print(f"\n=== Training {agent_name} with {policy_type} policy ===")
    
    # Get state and action dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent with appropriate configuration
    if agent_class == MaxEntAgent or agent_class == FisherMaxEntAgent:
        agent = agent_class(
            policy_type=policy_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['hidden_dims'],
            lr=config['lr'],
            temperature=config['temperature'],
            gamma=config['gamma']
        )
    elif agent_class == PPOAgent:
        agent = agent_class(
            policy_type=policy_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['hidden_dims'],
            lr=config['lr'],
            epsilon=config['epsilon'],
            gamma=config['gamma']
        )
    elif agent_class == TRPOAgent:
        agent = agent_class(
            policy_type=policy_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['hidden_dims'],
            gamma=config['gamma'],
            kl_delta=config['kl_delta']
        )
    
    # Train the agent
    rewards, losses, metrics = agent.train(
        env=env,
        num_episodes=config['num_episodes'],
        max_steps=config['max_steps']
    )
    
    # Save agent model
    model_path = f"results/{agent_name}_{policy_type}.pt"
    torch.save({
        'policy_type': policy_type,
        'policy_state_dict': agent.policy_net.state_dict(),
        'rewards': rewards,
        'losses': losses,
        'metrics': metrics
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    return agent, (rewards, losses, metrics)

def run_simulation(config):
    """
    Run a full simulation comparing different agents on the same environment
    
    Args:
        config: Configuration dictionary
    """
    print("Starting reinforcement learning agent comparison...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = GridWorld(
        grid_size=config['grid_size'],
        stochastic=config['stochastic'],
        noise=config['noise']
    )
    
    # Define agents to compare
    agent_classes = {
        "MaxEnt": MaxEntAgent,
        "FisherMaxEnt": FisherMaxEntAgent,
        "PPO": PPOAgent,
        "TRPO": TRPOAgent
    }
    
    policy_types = config['policy_types']
    
    # Train each agent with each policy type
    all_agents = {}
    all_metrics = {}
    
    for agent_name, agent_class in agent_classes.items():
        for policy_type in policy_types:
            full_name = f"{agent_name}_{policy_type}"
            agent, metrics = train_agent(
                agent_class, 
                agent_name, 
                policy_type, 
                env, 
                config
            )
            all_agents[full_name] = agent
            all_metrics[full_name] = metrics
    
    # Compare training curves
    compare_training_curves(
        all_metrics, 
        window=config['smoothing_window'],
        save_path="plots/training_comparison.png"
    )
    
    # Compare policies (for each policy type)
    for policy_type in policy_types:
        type_agents = {k: v for k, v in all_agents.items() if policy_type in k}
        compare_policy_visualizations(
            env, 
            type_agents, 
            save_path=f"plots/policy_comparison_{policy_type}.png"
        )
    
    # Compare trajectories (for each policy type)
    for policy_type in policy_types:
        type_agents = {k: v for k, v in all_agents.items() if policy_type in k}
        compare_trajectories(
            env, 
            type_agents, 
            num_trajectories=config['viz_trajectories'],
            max_steps=config['max_steps'],
            save_path=f"plots/trajectory_comparison_{policy_type}.png"
        )
    
    # Compare state visitation (for each policy type)
    for policy_type in policy_types:
        type_agents = {k: v for k, v in all_agents.items() if policy_type in k}
        compare_visitation_heatmaps(
            env, 
            type_agents, 
            num_episodes=config['viz_episodes'],
            max_steps=config['max_steps'],
            save_path=f"plots/visitation_comparison_{policy_type}.png"
        )
    
    print("Simulation complete! All plots saved to 'plots/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RL agent comparison simulation')
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, default=10, help='Size of the grid')
    parser.add_argument('--stochastic', action='store_true', help='Make environment stochastic')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level for stochastic environment')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Agent-specific parameters
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for MaxEnt agents')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Clipping parameter for PPO')
    parser.add_argument('--kl_delta', type=float, default=0.01, help='KL constraint for TRPO')
    
    # Network parameters
    parser.add_argument('--hidden_dims', type=str, default='[64, 64]', 
                       help='Hidden dimensions as a list (e.g., [64, 64])')
    parser.add_argument('--policy_types', type=str, default='mlp', 
                       choices=['mlp', 'rnn', 'transformer'], 
                       help='Policy network types to use')
    
    # Visualization parameters
    parser.add_argument('--smoothing_window', type=int, default=50, 
                       help='Window size for smoothing training curves')
    parser.add_argument('--viz_trajectories', type=int, default=5, 
                       help='Number of trajectories to visualize')
    parser.add_argument('--viz_episodes', type=int, default=50, 
                       help='Number of episodes for state visitation visualization')
    
    args = parser.parse_args()
    
    # Process arguments
    config = {
        # Environment parameters
        'grid_size': args.grid_size,
        'stochastic': args.stochastic,
        'noise': args.noise,
        
        # Training parameters
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'lr': args.lr,
        'gamma': args.gamma,
        
        # Agent-specific parameters
        'temperature': args.temperature,
        'epsilon': args.epsilon,
        'kl_delta': args.kl_delta,
        
        # Network parameters
        'hidden_dims': eval(args.hidden_dims),  # Convert string to list
        'policy_types': [args.policy_types],  # Convert to list
        
        # Visualization parameters
        'smoothing_window': args.smoothing_window,
        'viz_trajectories': args.viz_trajectories,
        'viz_episodes': args.viz_episodes
    }
    
    run_simulation(config)