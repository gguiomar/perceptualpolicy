# run_avoidance_task.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the environment
from envs.active_avoidance_env_test import ActiveAvoidanceEnv2D

# Import agents
from agents.ppo_agent import PPOAgent
from agents.maxent_agent import MaxEntAgent, FisherMaxEntAgent
from agents.trpo_agent import TRPOAgent

# Import the visualization functions
from utils.avoidance_visualization import (
    plot_avoidance_training_curves,
    plot_avoidance_trajectories_2d,
    plot_avoidance_heatmap_2d,
    animate_avoidance_trajectories_2d
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# --- Configuration ---
config = {
    # Environment params
    'height': 10,
    'width': 20,
    'tone_duration_steps': 10,
    'shock_delay_steps': 10,
    'max_steps_per_episode': 100,
    'initial_task': ActiveAvoidanceEnv2D.AVOID_TONE_1,

    # Agent params
    'agent_class': MaxEntAgent, 
    'agent_name': 'MaxEnt_RNN', 
    'policy_type': 'rnn',       
    'hidden_dim': 128,           # Hidden dimension for RNN
    # 'hidden_dims': [128, 128], # multi-layer MLP - comment out for rnn
    'lr': 0.001,                
    'gamma': 0.993,
    # 'epsilon': 0.2,           # PPO specific
    'temperature': 0.15,         # MaxEnt specific
    # 'kl_delta': 0.01,         # TRPO specific
    'rnn_type': 'gru',          # Specify RNN type ('gru' or 'lstm' or 'rnn')

    # Training params
    'num_episodes': 2000,
    'task_switch_episode': 1000
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ActiveAvoidanceEnv2D(
    height=config['height'],
    width=config['width'],
    tone_duration_steps=config['tone_duration_steps'],
    shock_delay_steps=config['shock_delay_steps'],
    max_steps_per_episode=config['max_steps_per_episode'],
    initial_task=config['initial_task']
)

state_dim = env.observation_space.shape[0] # Should be 4
action_dim = env.action_space.n

# --- Create Agent ---
AgentClass = config['agent_class']

# Construct agent parameters
agent_params = {
    'policy_type': config['policy_type'],
    'state_dim': state_dim,
    'action_dim': action_dim,
    'lr': config['lr'],
    'gamma': config['gamma']
}
if config['policy_type'] == 'mlp':
    agent_params['hidden_dims'] = config.get('hidden_dims', [128, 128])
elif config['policy_type'] in ['rnn', 'transformer']:
    agent_params['hidden_dim'] = config['hidden_dim']
    if config['policy_type'] == 'rnn':
         agent_params['rnn_type'] = config.get('rnn_type', 'gru')

# Add algorithm-specific params
if AgentClass == PPOAgent:
    agent_params['epsilon'] = config['epsilon']
elif AgentClass in [MaxEntAgent, FisherMaxEntAgent]:
     agent_params['temperature'] = config['temperature']
     # Add Fisher specific params if needed for FisherMaxEntAgent
elif AgentClass == TRPOAgent:
     agent_params['kl_delta'] = config['kl_delta']
     # Add TRPO specific params if needed

# Instantiate the agent
agent = AgentClass(**agent_params)

print(f"Created {config['agent_name']} with {config['policy_type']} policy.")
print(f"State dim: {state_dim}, Action dim: {action_dim}")

# --- Training ---
print("Starting training...")
rewards_history = []
losses_history = []
metric_history = []
avoidance_rate_history = []
shock_rate_history = []

for episode in range(config['num_episodes']):
    # Task Switching Logic
    if 'task_switch_episode' in config and episode == config['task_switch_episode']:
        print(f"\n--- Switching Task at Episode {episode} ---")
        env.switch_task()

    # Run Episode
    state = env.reset()
    states, actions, rewards, log_probs_old_list = [], [], [], []
    hidden_states_list = []
    episode_reward = 0
    ep_info = {'avoided': False, 'shocked': False}

    hidden_state = None
    if config['policy_type'] in ["rnn"] and hasattr(agent.policy_net, 'init_hidden'):
         hidden_state = agent.policy_net.init_hidden().to(device)

    for step in range(config['max_steps_per_episode']):
        # Select action
        if config['policy_type'] in ["rnn"]:
            action, log_prob, _, h_new = agent.select_action(state, hidden_state)
            hidden_states_list.append(hidden_state)
            hidden_state = h_new
        else:
            action, log_prob, _ = agent.select_action(state)

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs_old_list.append(log_prob) # Used by PPO/TRPO, ignored by MaxEnt update

        episode_reward += reward
        state = next_state

        if done:
            ep_info = info
            break

    # Agent Update
    update_args = [states, actions, rewards, log_probs_old_list] # MaxEnt ignores log_probs_old_list
    if config['policy_type'] in ["rnn"]:
         update_args.append(hidden_states_list)

    update_info = agent.update(*update_args)

    # Record History
    rewards_history.append(episode_reward)
    losses_history.append(update_info.get('loss', update_info.get('policy_loss', 0)))
    # MaxEnt returns 'entropy', PPO/TRPO return 'approx_kl' or 'kl'
    metric_history.append(update_info.get('approx_kl', update_info.get('entropy', update_info.get('kl', 0))))
    avoidance_rate_history.append(1 if ep_info['avoided'] else 0)
    shock_rate_history.append(1 if ep_info['shocked'] else 0)

    # Print Progress
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(rewards_history[-50:])
        avg_loss = np.mean(losses_history[-50:])
        avg_metric = np.mean(metric_history[-50:])
        avg_avoid = np.mean(avoidance_rate_history[-50:]) * 100
        avg_shock = np.mean(shock_rate_history[-50:]) * 100

        metric_name = 'Entropy' if isinstance(agent, (MaxEntAgent, FisherMaxEntAgent)) else 'KL'
        print(f"Ep {episode+1}/{config['num_episodes']} | Task: {'Tone1' if env.current_task_id == 1 else 'Tone2'} | "
              f"Rwd: {avg_reward:.2f} | Loss: {avg_loss:.4f} | {metric_name}: {avg_metric:.4f} | "
              f"Avoid%: {avg_avoid:.1f} | Shock%: {avg_shock:.1f}")

print("Training finished.")

# --- Plotting Results & Visualization ---
print("Plotting results...")
metric_name = 'Entropy' if isinstance(agent, (MaxEntAgent, FisherMaxEntAgent)) else 'KL'

plot_avoidance_training_curves(
    rewards=rewards_history,
    losses=losses_history,
    metrics=metric_history,
    metric_name=metric_name,
    avoidance_rates=avoidance_rate_history,
    shock_rates=shock_rate_history,
    smooth_window=50,
    save_path=f"plots/{config['agent_name']}_training_curves.png",
)
task_id_final = env.current_task_id

print("Generating visualizations...")

try:
    save_name = f"plots/{config['agent_name']}_trajectory_task{task_id_final}.png"
    plot_avoidance_trajectories_2d(env, agent, num_trajectories=8, max_steps=config['max_steps_per_episode'], save_path=save_name)
except Exception as e:
    print(f"Could not generate trajectory plot: {e}")

# try:
#     save_name = f"plots/avoidance_heatmap_task{task_id_final}.png"
#     plot_avoidance_heatmap_2d(env, agent, num_episodes=100, max_steps=config['max_steps_per_episode'], save_path=save_name)
# except Exception as e:
#     print(f"Could not generate heatmap plot: {e}")

print("Run complete.")