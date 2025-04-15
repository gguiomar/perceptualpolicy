# envs/active_avoidance_env.py

import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

class DiscreteSpace:
    """Defines a discrete action space (e.g., 0, 1, 2, 3)."""
    def __init__(self, n):
        self.n = n # Number of discrete actions
    def sample(self):
        return np.random.randint(self.n)

class BoxSpace:
    """Defines a continuous state space."""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)   # Minimum value for each dimension
        self.high = np.array(high, dtype=dtype) # Maximum value for each dimension
        self.shape = shape                      # Shape of the observation array (e.g., (4,))
        self.dtype = dtype                      # Data type
    def sample(self):
        # Generate a random sample within the space bounds
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

class ActiveAvoidanceEnv2D:
    """
    2D Active Avoidance Task Environment.

    State: [norm_x, norm_y, tone1_on, tone2_on, norm_time_since_tone]
    Action: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (STAY)
    Reward: +10 for avoidance, -10 for shock, -0.1 per step
    """
    # Action definitions
    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
    # Map actions to coordinate changes (dx, dy)
    ACTION_MAP = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

    # Task definitions
    AVOID_TONE_1 = 1 # Requires crossing the vertical midline
    AVOID_TONE_2 = 2 # Requires crossing the horizontal midline

    def __init__(self, height=10, width=20, tone_duration_steps=50,
                 shock_delay_steps=50, max_steps_per_episode=100,
                 initial_task=AVOID_TONE_1):
        """
        Initializes the environment parameters.
        Args:
            height: Height of the grid.
            width: Width of the grid.
            tone_duration_steps: How many steps the tone cue lasts.
            shock_delay_steps: Steps until shock occurs if not avoided.
            max_steps_per_episode: Maximum steps per trial.
            initial_task: Task the environment starts with (AVOID_TONE_1 or AVOID_TONE_2).
        """
        self.height = height
        self.width = width
        # Calculate center coordinates for midline checks
        self.center_x = (width - 1) / 2.0
        self.center_y = (height - 1) / 2.0

        self.tone_duration_steps = tone_duration_steps
        # Ensure shock delay is at least as long as tone duration
        self.shock_delay_steps = max(shock_delay_steps, tone_duration_steps)
        self.max_steps_per_episode = max_steps_per_episode
        # Internal variable tracking the current active rule (not observed by the agent)
        self.current_task_id = initial_task

        # Define action space (5 discrete actions)
        self.action_space = DiscreteSpace(5)
        # Define observation space (5 continuous values, normalized)
        self.observation_space = BoxSpace(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(5,), # State is [norm_x, norm_y, tone1_on, tone2_on, norm_time]
            dtype=np.float32
        )

        # Internal variables tracking the current state of the environment
        self.agent_pos = [0, 0] # Agent's [x, y] coordinates
        self.tone1_on = 0.0     # Flag indicating if tone 1 is active
        self.tone2_on = 0.0     # Flag indicating if tone 2 is active
        self.time_since_tone_onset = 0 # Step counter since tone started
        self.steps_in_episode = 0      # Step counter for the current episode
        self.current_zone = (0, 0) # Agent's current zone relative to midlines (x_zone, y_zone)

        # Variables for potential compatibility with visualization.py
        self.start = (0,0)
        self.goal = (-1,-1) # No fixed goal state
        self.obstacles = [] # No obstacles in this version

    def _get_state(self):
        """Calculates and returns the normalized state vector for the agent."""
        norm_x = self.agent_pos[0] / (self.width - 1)
        norm_y = self.agent_pos[1] / (self.height - 1)
        # Normalize time elapsed since tone started
        norm_time = min(self.time_since_tone_onset / self.tone_duration_steps, 1.0)
        # Return the state vector with both tones
        return np.array([norm_x, norm_y, self.tone1_on, self.tone2_on, norm_time], dtype=np.float32)

    def _get_zone(self, pos):
        """Determines the zone (0=left/bottom, 1=right/top) based on position."""
        x_zone = 0 if pos[0] <= self.center_x else 1
        y_zone = 0 if pos[1] <= self.center_y else 1
        return (x_zone, y_zone)

    def reset(self):
        """Resets the environment for a new episode (trial)."""
        self.steps_in_episode = 0
        self.tone1_on = 0.0
        self.tone2_on = 0.0
        self.time_since_tone_onset = 0

        # Determine starting position randomly, ensuring it's on the correct
        # side of the relevant midline for the current task.
        start_x, start_y = 0, 0

        if np.random.rand() < 0.5: # Start left half
            start_x = np.random.randint(0, int(np.floor(self.center_x)) + 1)
        else: # Start right half
            start_x = np.random.randint(int(np.ceil(self.center_x)), self.width)
        start_y = np.random.randint(0, self.height)

        self.agent_pos = [start_x, start_y]
        self.start = tuple(self.agent_pos) # Store start pos
        self.current_zone = self._get_zone(self.agent_pos) # Update current zone

        # Schedule tone to start in the first half of the trial
        self.tone_start_step = np.random.randint(0, self.max_steps_per_episode // 2)

        # Return the initial state observation
        return self._get_state()

    def switch_task(self, new_task_id=None):
        """Switches the current task rule (X-shuttle vs Y-shuttle)."""
        if new_task_id: # Allow setting a specific task
            if new_task_id not in [self.AVOID_TONE_1, self.AVOID_TONE_2]:
                raise ValueError("Invalid task ID")
            self.current_task_id = new_task_id
        else: # Default: toggle between Task 1 and Task 2
            self.current_task_id = self.AVOID_TONE_2 if self.current_task_id == self.AVOID_TONE_1 else self.AVOID_TONE_1
        print(f"Switched to Task {self.current_task_id} ({'Tone1' if self.current_task_id == 1 else 'Tone2'}-Shuttle)")

    def step(self, action):
        """Executes one time step in the environment."""
        if not (0 <= action < 5):
            raise ValueError(f"Invalid action: {action}")

        reward = -0.1  # step cost
        done = False   # Flag indicating if the episode ended this step
        info = {
            'shocked': False, 
            'avoided': False,
            'task': self.current_task_id,
            'tone1_active': self.tone1_on > 0.5,
            'tone2_active': self.tone2_on > 0.5
        }

        # Calculate agent's next potential position based on action
        dx, dy = self.ACTION_MAP[action]
        next_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]

        # Ensure the agent stays within the grid boundaries
        next_pos[0] = np.clip(next_pos[0], 0, self.width - 1)
        next_pos[1] = np.clip(next_pos[1], 0, self.height - 1)

        # Update the agent's position
        self.agent_pos = next_pos

        # --- Tone logic: activate tone in the first half of the trial ---
        if self.steps_in_episode == self.tone_start_step:
            if np.random.rand() < 0.5:
                self.tone1_on = 1.0
                self.tone2_on = 0.0
            else:
                self.tone1_on = 0.0
                self.tone2_on = 1.0

        avoided = False
        new_zone = self._get_zone(self.agent_pos) # Check zone after moving
        # --- Main trial logic: applies only if either tone is active ---
        if self.tone1_on > 0.5 or self.tone2_on > 0.5:
            self.time_since_tone_onset += 1 # Increment tone timer
            # Check if the agent performed the correct avoidance action for the CURRENT task
            if self.current_task_id == self.AVOID_TONE_1: # Task 1 rule
                ##TODO: This condiiton needs to be checked if aligned with real experiment
                if (self.tone1_on > 0.5 and self.current_zone[0] != new_zone[0]) or (self.tone2_on > 0.5 and self.time_since_tone_onset >= self.shock_delay_steps):
                    avoided = True
            elif self.current_task_id == self.AVOID_TONE_2: # Task 2 rule
                ##TODO: This condiiton needs to be checked if aligned with real experiment
                if (self.tone2_on > 0.5 and self.current_zone[0] != new_zone[0]) or (self.tone1_on > 0.5 and self.time_since_tone_onset >= self.shock_delay_steps):
                    avoided = True

        ##TODO: Current setup the agent gets reward for ignoring the tone
        if avoided:
            reward += 10.0 # Positive reward
            done = True    # End the episode
            info['avoided'] = True
            # Turn off both tones
            self.tone1_on = 0.0
            self.tone2_on = 0.0
            
        # If not avoided, check if the shock timer has expired
        if self.time_since_tone_onset >= self.shock_delay_steps and not avoided:
            # Shock occurs
            reward -= 10.0 # Negative reward
            if self.current_zone[0] != new_zone[0]:
                done = True    # End the episode
            info['shocked'] = True
            # Turn off both tones
            self.tone1_on = 0.0
            self.tone2_on = 0.0

            # Update the agent's current zone if the episode is still ongoing
            if not done:
                self.current_zone = new_zone

        # --- Check if max steps reached ---
        self.steps_in_episode += 1
        if self.steps_in_episode >= self.max_steps_per_episode:
            done = True # End episode
            # Apply timeout penalty only if not already ended by avoidance/shock
            if not info['avoided'] and not info['shocked']:
                reward -= 5.0

        # Get the state observation for the next step
        next_state = self._get_state()

        # Return step results
        return next_state, reward, done, info

    def render(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            show = True
        else:
            show = False
            ax.clear()

        # Mark agent position
        ax.scatter(self.agent_pos[0], self.agent_pos[1], marker='o', s=100, color='blue', label='Agent')

        # Draw center lines
        ax.axvline(self.center_x, color='gray', linestyle='--', lw=1)
        ax.axhline(self.center_y, color='gray', linestyle='--', lw=1)

        # Highlight the currently active boundary based on the task rule
        if self.current_task_id == self.AVOID_TONE_1:
             ax.axvline(self.center_x, color='red', linestyle='-', lw=2, label='X-Shuttle Boundary')
        else:
             ax.axhline(self.center_y, color='red', linestyle='-', lw=2, label='Y-Shuttle Boundary')

        # Setup grid appearance
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal', adjustable='box')

        # Display current step, tone status, and task info
        tone1_str = "ON" if self.tone1_on > 0.5 else "OFF"
        tone2_str = "ON" if self.tone2_on > 0.5 else "OFF"
        time_str = f"{self.time_since_tone_onset}/{self.tone_duration_steps}"
        task_str = f"Task: {'X' if self.current_task_id == 1 else 'Y'}-Shuttle"
        title = f"Step: {self.steps_in_episode} | Tone1: {tone1_str} | Tone2: {tone2_str} ({time_str}) | {task_str}"
        ax.set_title(title)
        ax.legend(loc='upper right')

        if show:
            plt.show()
