# run_avoidance_task.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict
import pandas as pd

# Import the environment
from envs.active_avoidance_env import ActiveAvoidanceEnv2D

# Import agents
#from agents.ppo_agent import PPOAgent
from agents.maxent_agent import MaxEntAgent, FisherMaxEntAgent
#from agents.trpo_agent import TRPOAgent

# Import the visualization functions
from utils.avoidance_visualization import (
    plot_avoidance_training_curves,
    plot_avoidance_trajectories_2d,
    plot_avoidance_heatmap_2d,
    plot_avoidance_trajectory_step_by_step,
    plot_multiple_avoidance_trajectories,
    plot_dual_task_performance,
    plot_loss_components
)

# --- After training: visualize weight changes over time ---
import matplotlib.animation as animation

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)
os.makedirs("plots/weights", exist_ok=True)

# --- Configuration ---
config = {
    # Environment params
    'height': 3,
    'width': 6,
    'max_tone_duration_steps': 6, # Max duration tone plays if not avoided
    'shock_onset_delay_steps': 6, # Steps after tone onset before shock starts
    'max_steps_per_episode': 12,
    'initial_task': ActiveAvoidanceEnv2D.AVOID_TONE_1,
    'move_penalty': -0.05,
    'shock_penalty_per_step': -1.0,
    'avoidance_reward': 0.3,

    # Agent params
    'agent_class': MaxEntAgent, # Choose between PPOAgent, MaxEntAgent, FisherMaxEntAgent, TRPOAgent
    'agent_name': 'MaxEntAgent_GRU_adap_temp',
    'policy_type': 'rnn',
    'hidden_dim': 128,           # Hidden dimension for RNN
    # 'hidden_dims': [128, 128], # multi-layer MLP - comment out for rnn
    'lr': 0.001,
    'gamma': 0.997, 
    # 'epsilon': 0.2,           # PPO specific
    # 'kl_delta': 0.01,         # TRPO specific
    'rnn_type': 'gru',          # Specify RNN type ('gru' or 'lstm' or 'rnn')
    'gradient_clip_norm': 1.0,

# --- Adaptive Temperature Params ---
    'use_adaptive_temp': True,       # Enable adaptive temperature
    'initial_temperature': 0.1,      # Starting temperature (higher for exploration)
    'min_temperature': 0.01,         # Minimum temperature (for exploitation)
    'max_temperature': 0.3,          # Maximum temperature (cap exploration)
    # Define reward range for scaling temperature (adjust based on expected rewards)
    'reward_range_for_temp': (-5.0, 0.1), # Expected min/max smoothed reward
    'adaptive_temp_window': 50,      # Window size for averaging reward

    # Training params
    'num_episodes': 20000,
    'task_switch_episode': 10000,
    'log_interval': 20, # How often to print progress
    'hidden_state_sampling_rate': 50,  # Sample every N episodes for visualization
    'weights_sampling_rate': 50 # How often to sample weights
}
# --- Use initial temperature if adaptive is disabled ---
if not config.get('use_adaptive_temp', False):
    config['temperature'] = config.get('initial_temperature', 0.1)
else:
     # Set initial temp for agent constructor, it will be adapted later
     config['temperature'] = config.get('initial_temperature', 0.1)
# -------------------------------------------------------


# Check for MPS (Mac GPU) first, then CUDA, then fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Environment Setup ---
env = ActiveAvoidanceEnv2D(
    height=config['height'],
    width=config['width'],
    max_tone_duration_steps=config['max_tone_duration_steps'],
    shock_onset_delay_steps=config['shock_onset_delay_steps'],
    max_steps_per_episode=config['max_steps_per_episode'],
    initial_task=config['initial_task'],
    move_penalty=config['move_penalty'],
    shock_penalty_per_step=config['shock_penalty_per_step'],
    avoidance_reward=config['avoidance_reward']
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# --- Track weights ---
policy_weights_over_time = []
gru_input_weights_over_time = []

# --- Create Agent ---
AgentClass = config['agent_class']
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

# PPO and TRPO removed for now
# if AgentClass == PPOAgent:
#     agent_params['epsilon'] = config['epsilon']
# elif AgentClass in [MaxEntAgent, FisherMaxEntAgent]:
#      # Pass the *initial* temperature
#      agent_params['temperature'] = config['temperature']
#      agent_params['gradient_clip_norm'] = config.get('gradient_clip_norm', None)
#      if AgentClass == FisherMaxEntAgent:
#           agent_params['use_natural_gradient'] = config.get('use_natural_gradient', True)
#           agent_params['cg_iters'] = config.get('cg_iters', 10)
#           agent_params['cg_damping'] = config.get('cg_damping', 1e-2)
# elif AgentClass == TRPOAgent:
#      agent_params['kl_delta'] = config['kl_delta']
# Pass the *initial* temperature
#      agent_params['temperature'] = config['temperature']
#      agent_params['gradient_clip_norm'] = config.get('gradient_clip_norm', None)
#      if AgentClass == FisherMaxEntAgent:
#           agent_params['use_natural_gradient'] = config.get('use_natural_gradient', True)
#           agent_params['cg_iters'] = config.get('cg_iters', 10)
#           agent_params['cg_damping'] = config.get('cg_damping', 1e-2)
agent = AgentClass(**agent_params)

print(f"Created {config['agent_name']} with {config['policy_type']} policy.")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"Agent parameters: {agent_params}")
print(f"Using Adaptive Temperature: {config.get('use_adaptive_temp', False)}")

# --- Training ---
print("Starting training...")
history = defaultdict(list)
# Use pandas Series for efficient rolling window calculation
reward_series = pd.Series(dtype=np.float64)

# Add hidden state collection
hidden_states_history = []
episode_hidden_states = []
episode_indices = []  # Track which episodes we're sampling from

for episode in range(config['num_episodes']):
    current_task_id = env.current_task_id

    if 'task_switch_episode' in config and episode == config['task_switch_episode']:
        print(f"\n--- Switching Task at Episode {episode} ---")
        env.switch_task()
        current_task_id = env.current_task_id

        # Save weight heatmap at task switch
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            W_policy = agent.policy_net.fc.weight.detach().cpu().numpy()
            im = ax.imshow(W_policy, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0)
            plt.colorbar(im, ax=ax, label='Weight Value')
            action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay']
            ax.set_yticks(range(len(action_labels)))
            ax.set_yticklabels(action_labels)
            ax.set_xlabel("GRU Hidden Units")
            ax.set_ylabel("Action")
            ax.set_title(f"Policy Weights at Task Switch (Episode {episode})")
            plt.savefig(f"plots/weights/{config['agent_name']}_policy_weights_task_switch.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not save task switch weight heatmap: {e}")

    # --- Adaptive Temperature Update ---
    if config.get('use_adaptive_temp', False) and episode >= config['adaptive_temp_window']:
        # Calculate smoothed reward over the window
        avg_reward_adaptive = reward_series.rolling(window=config['adaptive_temp_window']).mean().iloc[-1]

        # Linear scaling from reward range to temperature range
        min_r, max_r = config['reward_range_for_temp']
        min_t, max_t = config['min_temperature'], config['max_temperature']

        # Normalize reward (clamp between 0 and 1)
        norm_reward = max(0.0, min(1.0, (avg_reward_adaptive - min_r) / (max_r - min_r + 1e-8))) # Add epsilon

        # Calculate new temperature (higher reward -> lower temp)
        new_temp = max_t - norm_reward * (max_t - min_t)

        # Clamp temperature to defined bounds
        new_temp = max(min_t, min(max_t, new_temp))

        # Update agent's temperature
        if hasattr(agent, 'set_temperature'):
             agent.set_temperature(new_temp)
        else:
             # Fallback if method doesn't exist (should not happen with MaxEntAgent)
             agent.temperature = new_temp
    # ------------------------------------

    state = env.reset()
    # (Rest of the episode loop: data collection)
    states, actions, rewards, log_probs_old_list = [], [], [], []
    hidden_states_list = []
    episode_reward = 0
    ep_info = {}
    hidden_state = None
    if config['policy_type'] in ["rnn"] and hasattr(agent.policy_net, 'init_hidden'):
         hidden_state = agent.policy_net.init_hidden().to(device)

    for step in range(config['max_steps_per_episode']):
        if config['policy_type'] in ["rnn"]:
            # Handle state tensor conversion properly
            if isinstance(state, torch.Tensor):
                state_tensor = state.to(device)
            else:
                state_tensor = torch.FloatTensor(state).to(device)
            action, log_prob, _, h_new = agent.select_action(state_tensor, hidden_state)
            hidden_states_list.append(hidden_state)
            hidden_state = h_new
        else:
            action, log_prob, _ = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs_old_list.append(log_prob)
        episode_reward += reward
        state = next_state
        if done:
            ep_info = info
            break
    else:
        ep_info = info

    # Store episode's final hidden state
    if config['policy_type'] in ["rnn"] and episode % config['hidden_state_sampling_rate'] == 0:
        # Reshape the hidden state to 2D before storing
        hidden_state_flat = hidden_state.detach().cpu().numpy().reshape(-1)
        episode_hidden_states.append(hidden_state_flat)
        episode_indices.append(episode)

    # Agent Update (will use the potentially updated temperature)
    update_args = [states, actions, rewards, log_probs_old_list]
    if config['policy_type'] in ["rnn"]:
         update_args.append(hidden_states_list)
    update_info = agent.update(*update_args)

    # --- Track weights every 50 episodes ---
    if episode % config['weights_sampling_rate'] == 0:
        # Track policy head weights (GRU → action logits)
        try:
            W_policy = agent.policy_net.fc.weight.detach().cpu().numpy().copy()
            policy_weights_over_time.append(W_policy)
        except Exception as e:
            print(f"[Warning] Could not track policy weights at episode {episode}: {e}")

        # Track GRU input weights (input -> GRU gates)
        try:
            W_ih = agent.policy_net.rnn.weight_ih_l0.detach().cpu().numpy().copy()
            W_reset, W_update, W_new = np.split(W_ih, 3, axis=0)
            gru_input_weights_over_time.append({
                'reset': W_reset,
                'update': W_update,
                'new': W_new
            })
        except Exception as e:
            print(f"[Warning] Could not track GRU input weights at episode {episode}: {e}")

    # Save weight heatmap at last episode
    if episode == config['num_episodes'] - 1:
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            W_policy = agent.policy_net.fc.weight.detach().cpu().numpy()
            im = ax.imshow(W_policy, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0)
            plt.colorbar(im, ax=ax, label='Weight Value')
            action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay']
            ax.set_yticks(range(len(action_labels)))
            ax.set_yticklabels(action_labels)
            ax.set_xlabel("GRU Hidden Units")
            ax.set_ylabel("Action")
            ax.set_title(f"Policy Weights at Final Episode {episode}")
            plt.savefig(f"plots/weights/{config['agent_name']}_policy_weights_final.png")
            plt.close()
        except Exception as e:
            print(f"[Warning] Could not save final weight heatmap: {e}")

    # Record History
    history['episode'].append(episode)
    history['reward'].append(episode_reward)
    history['loss'].append(update_info.get('policy_loss', np.nan))
    history['pg_loss'].append(update_info.get('pg_loss', np.nan))
    history['entropy_loss'].append(update_info.get('entropy_loss', np.nan))
    history['metric'].append(update_info.get('entropy', np.nan))
    history['task_id'].append(current_task_id)
    history['presented_tone'].append(ep_info.get('presented_tone', 0))
    history['is_relevant'].append(ep_info.get('is_relevant_tone', False))
    history['avoided'].append(1 if ep_info.get('avoided', False) else 0)
    history['shocked'].append(1 if ep_info.get('shocked', False) else 0)
    history['escaped'].append(1 if ep_info.get('escaped', False) else 0)
    history['shuttled'].append(1 if ep_info.get('shuttled', False) else 0)
    # Store the temperature used during this update
    history['temperature'].append(agent.temperature)

    # Update reward series for adaptive temp calculation
    # Use pd.concat instead of append for efficiency with larger datasets
    reward_series = pd.concat([reward_series, pd.Series([episode_reward])], ignore_index=True)


    # Print Progress
    log_interval = config['log_interval']
    if (episode + 1) % log_interval == 0:
        # (Keep the conditional stats calculation)
        recent_indices = range(max(0, episode + 1 - log_interval), episode + 1)
        task1_relevant_trials = [i for i in recent_indices if history['task_id'][i] == 1 and history['presented_tone'][i] == 1]
        task1_irrelevant_trials = [i for i in recent_indices if history['task_id'][i] == 1 and history['presented_tone'][i] == 2]
        task2_relevant_trials = [i for i in recent_indices if history['task_id'][i] == 2 and history['presented_tone'][i] == 2]
        task2_irrelevant_trials = [i for i in recent_indices if history['task_id'][i] == 2 and history['presented_tone'][i] == 1]

        def safe_mean(indices, key):
            if not indices: return 0.0
            values = [history[key][i] for i in indices if i < len(history[key])]
            if not values: return 0.0
            return np.mean(values) * 100

        t1_rel_avoid = safe_mean(task1_relevant_trials, 'avoided')
        t1_rel_shock = safe_mean(task1_relevant_trials, 'shocked')
        t1_rel_escape = safe_mean(task1_relevant_trials, 'escaped')
        t1_rel_shuttle = safe_mean(task1_relevant_trials, 'shuttled')
        t1_irrel_shuttle = safe_mean(task1_irrelevant_trials, 'shuttled')
        t2_rel_avoid = safe_mean(task2_relevant_trials, 'avoided')
        t2_rel_shock = safe_mean(task2_relevant_trials, 'shocked')
        t2_rel_escape = safe_mean(task2_relevant_trials, 'escaped')
        t2_rel_shuttle = safe_mean(task2_relevant_trials, 'shuttled')
        t2_irrel_shuttle = safe_mean(task2_irrelevant_trials, 'shuttled')

        avg_reward = np.mean(history['reward'][-log_interval:])
        valid_losses = [l for l in history['loss'][-log_interval:] if l is not None and not np.isnan(l)]
        avg_loss = np.mean(valid_losses) if valid_losses else np.nan
        valid_metrics = [m for m in history['metric'][-log_interval:] if m is not None and not np.isnan(m)]
        avg_metric = np.mean(valid_metrics) if valid_metrics else np.nan
        valid_pg_losses = [l for l in history['pg_loss'][-log_interval:] if l is not None and not np.isnan(l)]
        avg_pg_loss = np.mean(valid_pg_losses) if valid_pg_losses else np.nan
        valid_ent_losses = [l for l in history['entropy_loss'][-log_interval:] if l is not None and not np.isnan(l)]
        avg_ent_loss = np.mean(valid_ent_losses) if valid_ent_losses else np.nan
        current_temp = history['temperature'][-1] # Get the latest temperature

        # --- Updated Print Statement ---
        print(f"\nEp {episode+1}/{config['num_episodes']} | Temp: {current_temp:.3f} | Avg Rwd: {avg_reward:.2f} | Avg Loss: {avg_loss:.4f} (PG:{avg_pg_loss:.4f}, Ent:{avg_ent_loss:.4f}) | Avg Entropy: {avg_metric:.4f}")
        # -----------------------------
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

# --- Policy head animation ---
"""
Interpretation: 
    - Red indicates positive weights (encouraging the action),
    - Blue indicates negative weights (discouraging the action).
    - Stronger colors represent stronger connections between hidden units and actions.
"""
try:
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(policy_weights_over_time[0], aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0)
    plt.colorbar(im, ax=ax)
    
    # Add action labels
    action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay']  # Matches environment's action mapping: UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4
    ax.set_yticks(range(len(action_labels)))
    ax.set_yticklabels(action_labels)
    
    # Add x-axis labels for GRU units
    ax.set_xlabel("GRU Hidden Units (0-127)")
    ax.set_ylabel("Action")
    
    
    ax.set_title("Policy Head Weights (GRU → Action)")

    def update_policy(i):
        im.set_array(policy_weights_over_time[i])
        ax.set_title(f"Policy Weights at Episode {i*50}")
        return [im]

    ani = animation.FuncAnimation(fig, update_policy, frames=len(policy_weights_over_time), blit=True)
    ani.save(f"plots/weights/{config['agent_name']}_policy_weights_evolution.gif", writer='imagemagick', fps=30)
    plt.close()
    print("Saved policy head weight evolution animation.")
except Exception as e:
    print(f"[Error] Could not create policy weight animation: {e}")

# --- Plotting Results & Visualization ---
print("Plotting results...")
metric_name = 'Entropy'

# (Keep plotting calls as before)
plot_avoidance_training_curves(
    rewards=history['reward'],
    losses=history['loss'],
    metrics=history['metric'],
    metric_name=metric_name,
    avoidance_rates=[],
    shock_rates=[],
    smooth_window=50,
    save_path=f"plots/{config['agent_name']}_basic_training_curves.png",
)
plot_loss_components(
    history=history,
    smooth_window=50,
    save_path=f"plots/{config['agent_name']}_loss_components.png"
)
plot_dual_task_performance(
    history=history,
    task_switch_episode=config.get('task_switch_episode'),
    save_path=f"plots/{config['agent_name']}_dual_task_performance.png"
)

# Add plot for temperature history if adaptive temp used
if config.get('use_adaptive_temp', False):
     plt.figure(figsize=(10, 4))
     plt.plot(history['episode'], history['temperature'], label='Temperature')
     plt.xlabel('Episode')
     plt.ylabel('Temperature')
     plt.title('Adaptive Temperature Schedule')
     plt.grid(True)
     plt.legend()
     temp_plot_path = f"plots/{config['agent_name']}_temperature_schedule.png"
     plt.savefig(temp_plot_path)
     print(f"Temperature schedule plot saved to {temp_plot_path}")
     plt.close()


# (Keep trajectory visualization code as before)
task_id_final = env.current_task_id
print("Generating trajectory visualizations...")
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
            visualization_type='animation'
        )
    except Exception as e:
        print(f"Could not generate step-by-step trajectory animation example {i+1}: {e}")
print("Run complete.")

# After training, create hidden state visualizations
if config['policy_type'] in ["rnn"] and len(episode_hidden_states) > 0:
    print("\nGenerating hidden state visualizations...")
    
    # Convert hidden states to numpy array
    hidden_states_array = np.array(episode_hidden_states)
    print(f"Hidden states shape: {hidden_states_array.shape}")
    print(f"Sampling every {config['hidden_state_sampling_rate']} episodes")
    
    # Perform PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states_array)
    
    # Create discrete color map for tasks
    task_colors = ['blue', 'red']  # One color per task
    task_ids = [history['task_id'][i] for i in episode_indices]
    
    # Create PCA visualization
    plt.figure(figsize=(12, 8))
    for task_id in [1, 2]:  # Plot each task separately
        mask = [t == task_id for t in task_ids]
        plt.scatter(hidden_states_2d[mask, 0], hidden_states_2d[mask, 1], 
                   c=task_colors[task_id-1], label=f'Task {task_id}', alpha=0.6)
    
    plt.title('PCA of Hidden States During Training\n(Sampled every {} episodes)'.format(config['hidden_state_sampling_rate']))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(f"plots/{config['agent_name']}_hidden_states_pca.png")
    plt.close()
    
    # Create animation of hidden state evolution
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create separate scatter plots for each task
    scatter1 = ax.scatter([], [], c=task_colors[0], label='Task 1', alpha=0.6)
    scatter2 = ax.scatter([], [], c=task_colors[1], label='Task 2', alpha=0.6)
    
    ax.set_xlim(hidden_states_2d[:, 0].min() - 0.1, hidden_states_2d[:, 0].max() + 0.1)
    ax.set_ylim(hidden_states_2d[:, 1].min() - 0.1, hidden_states_2d[:, 1].max() + 0.1)
    ax.set_title('Evolution of Hidden States During Training\n(Sampled every {} episodes)'.format(config['hidden_state_sampling_rate']))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    
    def update(frame):
        # Update data for each task separately
        task1_mask = [t == 1 for t in task_ids[:frame]]
        task2_mask = [t == 2 for t in task_ids[:frame]]
        
        scatter1.set_offsets(hidden_states_2d[:frame][task1_mask])
        scatter2.set_offsets(hidden_states_2d[:frame][task2_mask])
        
        return scatter1, scatter2
    
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(hidden_states_2d), 
                        interval=50, blit=True)
    anim.save(f"plots/{config['agent_name']}_hidden_states_evolution.gif", 
              writer='pillow', fps=20)
    plt.close()