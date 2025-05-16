import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# TODO: the pca plot with the actions is not working corretly at the moment

# --- Configuration ---
MODEL_NAME = 'Agent_1'
RESULTS_DIR = 'results'
OUTPUT_DIR = 'plots'
DATA_FILE_PATH = os.path.join(RESULTS_DIR, 'training_data', f"{MODEL_NAME}_hidden_states_data.npz")

# --- Load Data ---
if not os.path.exists(DATA_FILE_PATH):
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    exit()

data = np.load(DATA_FILE_PATH, allow_pickle=True)
hidden_states = data['hidden_states'] # Shape: (n_samples, hidden_dim)
task_ids = data['task_ids']           # Shape: (n_samples,) e.g., [1, 1, 2, 1, 2, ...]
presented_tones = data['tones']       # Shape: (n_samples,) e.g., [0, 1, 0, 1, 0, ...]

# Load actions data
first_actions_for_samples = None
if 'actions' in data:
    actions_from_file = data['actions']
    if len(actions_from_file) == len(hidden_states):
        first_actions_for_samples = np.array([
            ep_actions[0] if isinstance(ep_actions, (list, np.ndarray)) and len(ep_actions) > 0 else -1
            for ep_actions in actions_from_file
        ])
    else:
        print(f"Warning: Mismatch in length of hidden_states ({len(hidden_states)}) and actions_from_file ({len(actions_from_file)}). Cannot reliably map actions.")
else:
    print(f"Warning: 'actions' key not found in {DATA_FILE_PATH}. Cannot create action-colored plot.")


# --- PCA ---
if hidden_states.shape[0] < 2:
    print("Error: Not enough samples for PCA.")
    exit()
pca = PCA(n_components=2)
pca_states = pca.fit_transform(hidden_states) # Shape: (n_samples, 2)

# --- Plotting Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

unique_tasks = np.unique(task_ids)
unique_tones = np.unique(presented_tones)

# Define shapes for tasks
task_markers = {task: marker for task, marker in zip(unique_tasks, ['o', '^', 's', 'D', 'v'][:len(unique_tasks)])}
# Define colors for tones
cmap_len_tones = len(unique_tones) if len(unique_tones) > 0 else 1
tone_colors_cmap = plt.cm.get_cmap('viridis', cmap_len_tones)
tone_colors = {tone: tone_colors_cmap(i) for i, tone in enumerate(unique_tones)}

ACTION_LABELS = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay', -1: 'N/A'}
cmap_len_actions = len(ACTION_LABELS) if len(ACTION_LABELS) > 0 else 1
action_colors_cmap = plt.cm.get_cmap('tab10', cmap_len_actions) # Using a qualitative colormap
action_plot_colors = {action_val: action_colors_cmap(i) for i, action_val in enumerate(sorted(ACTION_LABELS.keys()))}


# --- Plot 1
fig1, ax1 = plt.subplots(figsize=(10, 7))

for task_val in unique_tasks:
    for tone_val in unique_tones:
        mask = (task_ids == task_val) & (presented_tones == tone_val)
        if np.any(mask):
            ax1.scatter(
                pca_states[mask, 0],
                pca_states[mask, 1],
                color=tone_colors.get(tone_val, 'gray'), 
                marker=task_markers.get(task_val, 'x'),  
                alpha=0.7,
                s=50
            )

# Legend for Plot 1 (tone/task)
tone_legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'Tone {int(tone)}',
           markerfacecolor=tone_colors.get(tone, 'gray'), markersize=8)
    for tone in unique_tones
]
task_legend_elements_p1 = [
    Line2D([0], [0], marker=task_markers.get(task, 'x'), color='gray', label=f'Task {int(task)}',
           linestyle='None', markersize=8, markerfacecolor='gray')
    for task in unique_tasks
]
ax1.legend(handles=tone_legend_elements + task_legend_elements_p1, title="Legend", loc="best")

ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title(f'PCA: Color by Tone, Shape by Task ({MODEL_NAME})')
plt.grid(True)
plt.tight_layout()

output_filename_1 = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_pca_tone_color_task_shape.png")
plt.savefig(output_filename_1)
plt.close(fig1)
print(f"PCA plot (Tone/Task) saved to {output_filename_1}")


# --- Plot 2: Color by Action, Shape by Task ID ---
if first_actions_for_samples is not None:
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    
    unique_actions_in_data = np.unique(first_actions_for_samples)

    for task_val in unique_tasks:
        for action_val in unique_actions_in_data:
            mask = (task_ids == task_val) & (first_actions_for_samples == action_val)
            if np.any(mask):
                ax2.scatter(
                    pca_states[mask, 0],
                    pca_states[mask, 1],
                    color=action_plot_colors.get(action_val, 'black'), # Fallback color
                    marker=task_markers.get(task_val, 'x'),      # Fallback marker
                    alpha=0.7,
                    s=50
                )

    # Legend for Plot 2
    action_legend_elements = [
        Line2D([0], [0], marker='o', color='w', 
               label=ACTION_LABELS.get(action, f'Action {int(action)}'),
               markerfacecolor=action_plot_colors.get(action, 'black'), markersize=8)
        for action in unique_actions_in_data if action in ACTION_LABELS # Only for known actions
    ]
    task_legend_elements_p2 = [
        Line2D([0], [0], marker=task_markers.get(task, 'x'), color='gray', label=f'Task {int(task)}',
               linestyle='None', markersize=8, markerfacecolor='gray')
        for task in unique_tasks
    ]
    ax2.legend(handles=action_legend_elements + task_legend_elements_p2, title="Legend", loc="best")

    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title(f'PCA: Color by Action, Shape by Task ({MODEL_NAME})')
    plt.grid(True)
    plt.tight_layout()

    output_filename_2 = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_pca_action_color_task_shape.png")
    plt.savefig(output_filename_2)
    plt.close(fig2)
    print(f"PCA plot (Action/Task) saved to {output_filename_2}")
