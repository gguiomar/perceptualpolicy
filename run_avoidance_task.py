# run_avoidance_task.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Import the environment
from envs.active_avoidance_env import ActiveAvoidanceEnv2D

# Import agents
from agents.ppo_agent import PPOAgent
from agents.maxent_agent import MaxEntAgent, FisherMaxEntAgent
from agents.trpo_agent import TRPOAgent

# Import the visualization functions
from utils.avoidance_visualization import (
    plot_avoidance_training_curves,
    plot_avoidance_trajectories_2d,
    plot_avoidance_heatmap_2d,
    plot_avoidance_trajectory_step_by_step,
    plot_multiple_avoidance_trajectories,
    plot_dual_task_performance # Import the new function
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# --- Configuration ---
config = {
    # Environment params
    'height': 6,
    'width': 12,
    'max_tone_duration_steps': 15, # Max duration tone plays if not avoided
    'shock_onset_delay_steps': 15, # Steps after tone onset before shock starts
    'max_steps_per_episode': 100,
    'initial_task': ActiveAvoidanceEnv2D.AVOID_TONE_1,
    'move_penalty': -0.5,
    'shock_penalty_per_step': -1.0,
    'avoidance_reward': 0.02,

    # Agent params
    'agent_class': MaxEntAgent,
    'agent_name': 'MaxEnt_RNN',
    'policy_type': 'rnn',
    'hidden_dim': 128,           # Hidden dimension for RNN
    # 'hidden_dims': [128, 128], # multi-layer MLP - comment out for rnn
    'lr': 0.001,
    'gamma': 0.993,
    # 'epsilon': 0.2,           # PPO specific
    'temperature': 0.05,         # MaxEnt specific
    # 'kl_delta': 0.01,         # TRPO specific
    'rnn_type': 'gru',          # Specify RNN type ('gru' or 'lstm' or 'rnn')

    # Training params
    'num_episodes': 2000,
    'task_switch_episode': 1000,
    'log_interval': 20 # How often to print progress
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = ActiveAvoidanceEnv2D(
    height=config['height'],
    width=config['width'],
    max_tone_duration_steps=config['max_tone_duration_steps'],
    shock_onset_delay_steps=config['shock_onset_delay_steps'],
    max_steps_per_episode=config['max_steps_per_episode'],
    initial_task=config['initial_task'],
    move_penalty=config['move_penalty'],
    shock_penalty_per_step=config['shock_penalty_per_step'],
    # Pass avoidance reward to env constructor if it uses it
    avoidance_reward=config['avoidance_reward']
)

state_dim = env.observation_space.shape[0] 
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
# Use a dictionary to store history for better organization
history = defaultdict(list)

for episode in range(config['num_episodes']):
    current_task_id = env.current_task_id # Get task ID for this episode

    # Task Switching Logic
    if 'task_switch_episode' in config and episode == config['task_switch_episode']:
        print(f"\n--- Switching Task at Episode {episode} ---")
        env.switch_task()
        current_task_id = env.current_task_id # Update task ID after potential switch

    # Run Episode
    state = env.reset()
    states, actions, rewards, log_probs_old_list = [], [], [], []
    hidden_states_list = []
    episode_reward = 0
    ep_info = {} # Store final info dict

    hidden_state = None
    if config['policy_type'] in ["rnn"] and hasattr(agent.policy_net, 'init_hidden'):
         # Ensure hidden state is initialized on the correct device
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
            ep_info = info # Store the final info dict for the episode
            break
    else: # If loop finished without break (max steps reached)
        ep_info = info # Store the last info dict

    # Agent Update
    update_args = [states, actions, rewards, log_probs_old_list] # MaxEnt ignores log_probs_old_list
    if config['policy_type'] in ["rnn"]:
         update_args.append(hidden_states_list)

    update_info = agent.update(*update_args)

    # Record History (Stores raw 0/1 for each outcome per episode)
    history['episode'].append(episode)
    history['reward'].append(episode_reward)
    history['loss'].append(update_info.get('loss', update_info.get('policy_loss', np.nan))) # Use NaN if no loss
    history['metric'].append(update_info.get('approx_kl', update_info.get('entropy', update_info.get('kl', np.nan)))) # Use NaN if no metric
    history['task_id'].append(current_task_id)
    history['presented_tone'].append(ep_info.get('presented_tone', 0))
    history['is_relevant'].append(ep_info.get('is_relevant_tone', False))
    history['avoided'].append(1 if ep_info.get('avoided', False) else 0)
    history['shocked'].append(1 if ep_info.get('shocked', False) else 0)
    history['escaped'].append(1 if ep_info.get('escaped', False) else 0)
    history['shuttled'].append(1 if ep_info.get('shuttled', False) else 0)


    # Print Progress
    log_interval = config['log_interval']
    if (episode + 1) % log_interval == 0:
        # Calculate conditional stats *only* for the last 'log_interval' episodes for console display
        recent_indices = range(max(0, episode + 1 - log_interval), episode + 1)
        task1_relevant_trials = [i for i in recent_indices if history['task_id'][i] == 1 and history['presented_tone'][i] == 1]
        task1_irrelevant_trials = [i for i in recent_indices if history['task_id'][i] == 1 and history['presented_tone'][i] == 2]
        task2_relevant_trials = [i for i in recent_indices if history['task_id'][i] == 2 and history['presented_tone'][i] == 2]
        task2_irrelevant_trials = [i for i in recent_indices if history['task_id'][i] == 2 and history['presented_tone'][i] == 1]

        def safe_mean(indices, key):
            if not indices: return 0.0
            # Filter history based on indices before calculating mean
            values = [history[key][i] for i in indices if i < len(history[key])] # Ensure index exists
            if not values: return 0.0
            return np.mean(values) * 100

        # Performance for Task 1 context (averaged over log_interval)
        t1_rel_avoid = safe_mean(task1_relevant_trials, 'avoided')
        t1_rel_shock = safe_mean(task1_relevant_trials, 'shocked') # Shocked implies not avoided
        t1_rel_escape = safe_mean(task1_relevant_trials, 'escaped') # Escaped implies shocked
        t1_rel_shuttle = safe_mean(task1_relevant_trials, 'shuttled') # Avoided or Escaped
        t1_irrel_shuttle = safe_mean(task1_irrelevant_trials, 'shuttled') # Incorrect shuttle

        # Performance for Task 2 context (averaged over log_interval)
        t2_rel_avoid = safe_mean(task2_relevant_trials, 'avoided')
        t2_rel_shock = safe_mean(task2_relevant_trials, 'shocked')
        t2_rel_escape = safe_mean(task2_relevant_trials, 'escaped')
        t2_rel_shuttle = safe_mean(task2_relevant_trials, 'shuttled')
        t2_irrel_shuttle = safe_mean(task2_irrelevant_trials, 'shuttled')

        # Calculate overall averages over log_interval
        avg_reward = np.mean(history['reward'][-log_interval:])
        # Handle potential NaN values in loss/metric before averaging
        valid_losses = [l for l in history['loss'][-log_interval:] if l is not None and not np.isnan(l)]
        avg_loss = np.mean(valid_losses) if valid_losses else np.nan
        valid_metrics = [m for m in history['metric'][-log_interval:] if m is not None and not np.isnan(m)]
        avg_metric = np.mean(valid_metrics) if valid_metrics else np.nan
        metric_name = 'Entropy' if isinstance(agent, (MaxEntAgent, FisherMaxEntAgent)) else 'KL'

        print(f"\nEp {episode+1}/{config['num_episodes']} | Avg Rwd: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} | Avg {metric_name}: {avg_metric:.4f}")
        # Display results based on the task active 
        active_task_id_for_log = history['task_id'][episode]
        if active_task_id_for_log == 1:
             print(f"  Task 1 Active:")
             print(f"    Shock Trials    ({len(task1_relevant_trials)}): Avoid: {t1_rel_avoid:.1f}% | Shock: {t1_rel_shock:.1f}% | Escape: {t1_rel_escape:.1f}% | Shuttle: {t1_rel_shuttle:.1f}%")
             print(f"    No Shock Trials ({len(task1_irrelevant_trials)}): Incorrect Shuttle: {t1_irrel_shuttle:.1f}%")
        else: # Task 2 Active
             print(f"  Task 2 Active:")
             print(f"    Shock Trials    ({len(task2_relevant_trials)}): Avoid: {t2_rel_avoid:.1f}% | Shock: {t2_rel_shock:.1f}% | Escape: {t2_rel_escape:.1f}% | Shuttle: {t2_rel_shuttle:.1f}%")
             print(f"    No Shock Trials ({len(task2_irrelevant_trials)}): Incorrect Shuttle: {t2_irrel_shuttle:.1f}%")


print("Training finished.")

# --- Plotting Results & Visualization ---
print("Plotting results...")
metric_name = 'Entropy' if isinstance(agent, (MaxEntAgent, FisherMaxEntAgent)) else 'KL'

# Plot standard training curves (Reward, Loss, Metric)
plot_avoidance_training_curves(
    rewards=history['reward'],
    losses=history['loss'],
    metrics=history['metric'],
    metric_name=metric_name,
    avoidance_rates=[], # Pass empty lists as these rates are not plotted here
    shock_rates=[],
    smooth_window=50, # Example smoothing window for basic curves plot
    save_path=f"plots/{config['agent_name']}_basic_training_curves.png",
)

# Plot the dual-task performance curve
plot_dual_task_performance(
    history=history,
    task_switch_episode=config.get('task_switch_episode'),
    save_path=f"plots/{config['agent_name']}_dual_task_performance.png"
)


task_id_final = env.current_task_id
print("Generating trajectory visualizations...")

# Generate multiple example trajectories for the final task state
num_example_trajectories = 4
print(f"Generating {num_example_trajectories} example step-by-step trajectory plots/animations...")
for i in range(num_example_trajectories):
    try:
        base_save_name = f"plots/{config['agent_name']}_trajectory_example_{i+1}_task{task_id_final}"
        anim_save_path = base_save_name + ".gif"
        plot_avoidance_trajectory_step_by_step(
            env, agent,
            max_steps=config['max_steps_per_episode'],
            save_path=anim_save_path,
            visualization_type='animation' # Ensure animation is requested
        )
        # Optionally generate gallery as well
        # gallery_save_path = base_save_name + "_gallery" # Directory name
        # plot_avoidance_trajectory_step_by_step(
        #     env, agent,
        #     max_steps=config['max_steps_per_episode'],
        #     save_path=gallery_save_path, # Pass directory path
        #     visualization_type='gallery',
        #     gallery_save_dir=gallery_save_path # Specify directory explicitly
        # )
    except Exception as e:
        print(f"Could not generate step-by-step trajectory animation example {i+1}: {e}")
print("Run complete.")
