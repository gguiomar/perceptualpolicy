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
    2D Active Avoidance Task Environment (Updated Logic).

    State: [norm_x, norm_y, norm_trial_time, tone1_on, tone2_on]
    Action: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (STAY)
    """
    # Action definitions
    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
    # Map actions to coordinate changes (dx, dy)
    ACTION_MAP = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}

    # Task definitions
    AVOID_TONE_1 = 1 # Requires shuttling if tone 1 occurs
    AVOID_TONE_2 = 2 # Requires shuttling if tone 2 occurs

    def __init__(self, height=10, width=20, max_tone_duration_steps=20,
                 shock_onset_delay_steps=20, max_steps_per_episode=100,
                 initial_task=AVOID_TONE_1, move_penalty=-0.1,
                 shock_penalty_per_step=-1.0, avoidance_reward=0.1):
        """
        Initializes the environment parameters.
        Args:
            height: Height of the grid.
            width: Width of the grid (should be >= 3).
            max_tone_duration_steps: Max steps the tone cue lasts if not avoided.
            shock_onset_delay_steps: Steps from tone onset until shock starts if not avoided.
            max_steps_per_episode: Maximum steps per trial.
            initial_task: Task the environment starts with (AVOID_TONE_1 or AVOID_TONE_2).
            move_penalty: Penalty applied for taking a move action.
            shock_penalty_per_step: Penalty applied for each step during shock.
        """
        #if width < 3:
        #     raise ValueError("Width must be at least 3 for alternating start positions.")
        self.height = height
        self.width = width
        self.center_x = (width - 1) / 2.0

        self.max_tone_duration_steps = max_tone_duration_steps
        self.shock_onset_delay_steps = shock_onset_delay_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.move_penalty = move_penalty
        self.shock_penalty_per_step = shock_penalty_per_step
        self.avoidance_reward = avoidance_reward
        self.current_task_id = initial_task

        self.action_space = DiscreteSpace(5)
        self.observation_space = BoxSpace(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(5,), # State: [norm_x, norm_y, norm_trial_time, tone1_on, tone2_on]
            dtype=np.float32
        )

        # Internal state variables
        self.agent_pos = [0, 0]
        self.tone1_on = 0.0     # Indicates if tone 1 *sound* is currently playing
        self.tone2_on = 0.0     # Indicates if tone 2 *sound* is currently playing
        self.is_tone_active = False # Internal flag: is any tone logic active?
        self.is_shock_active = False # Internal flag: is shock currently being applied?
        self.time_since_tone_onset = 0
        self.steps_in_episode = 0
        self.current_zone = (0, 0) # Agent's current zone (0=left, 1=right) based on x-midline
        self.zone_at_tone_onset = (0, 0) # Zone when the relevant tone started
        self.active_tone_this_trial = 0 # Which tone (1 or 2) is chosen for this trial
        self.start_side_alternator = 0 # 0 for left start, 1 for right start
        self.shuttled_this_trial = False # Flag if agent crossed midline after tone onset

        self.start = (0,0)
        self.goal = (-1,-1)
        self.obstacles = []

    def _get_state(self):
        """Calculates and returns the normalized state vector for the agent."""
        norm_x = self.agent_pos[0] / (self.width-1)
        norm_y = self.agent_pos[1] / (self.height-1)
        norm_trial_time = min(self.steps_in_episode / self.max_steps_per_episode, 1.0)
        # Return the state vector including current tone sound status
        return np.array([norm_x, norm_y, norm_trial_time, self.tone1_on, self.tone2_on], dtype=np.float32)

    def _get_zone(self, pos):
        """Determines the x-zone (0=left, 1=right) based on position."""
        # Only x-midline matters for shuttling
        x_zone = 0 if pos[0] <= self.center_x else 1
        return (x_zone, 0) # Return tuple

    def reset(self):
        """Resets the environment for a new episode (trial)."""
        self.steps_in_episode = 0
        self.tone1_on = 0.0
        self.tone2_on = 0.0
        self.is_tone_active = False
        self.is_shock_active = False
        self.time_since_tone_onset = 0
        self.active_tone_this_trial = 0
        self.shuttled_this_trial = False # Reset shuttle flag

        # Determine starting position: Alternating x=1 and x=width-2
        if self.start_side_alternator == 0:
            start_x = 1
        else:
            start_x = self.width - 2
        self.start_side_alternator = 1 - self.start_side_alternator # Flip for next trial
        start_y = np.random.randint(0, self.height)

        self.agent_pos = [start_x, start_y]
        self.start = tuple(self.agent_pos)
        self.current_zone = self._get_zone(self.agent_pos)

        # Schedule tone: Randomly choose tone 1 or 2, schedule start time
        self.active_tone_this_trial = np.random.choice([1, 2])
        # Tone starts in the first half of the trial
        #self.tone_start_step = np.random.randint(0, self.max_steps_per_episode // 2)
        self.tone_start_step = 0
        return self._get_state()

    def switch_task(self, new_task_id=None):
        """Switches the current task rule (which tone is relevant for shock)."""
        if new_task_id:
            if new_task_id not in [self.AVOID_TONE_1, self.AVOID_TONE_2]:
                raise ValueError("Invalid task ID")
            self.current_task_id = new_task_id
        else:
            self.current_task_id = self.AVOID_TONE_2 if self.current_task_id == self.AVOID_TONE_1 else self.AVOID_TONE_1
        print(f"Switched to Task {self.current_task_id} ({'Tone1' if self.current_task_id == 1 else 'Tone2'}-Shuttle)")

    def step(self, action):
        """Executes one time step in the environment."""
        if not (0 <= action < 5):
            raise ValueError(f"Invalid action: {action}")

        reward = 0.0
        done = False
        # Initialize info dict for this step
        info = {
            'avoided': False,
            'shocked': False, # True if shock was ever active this trial
            'escaped': False, # True if agent escaped active shock
            'shuttled': self.shuttled_this_trial, # Carry over shuttle status
            'task': self.current_task_id,
            'presented_tone': self.active_tone_this_trial,
            'is_relevant_tone': False, # Will be set later if tone is active
            'tone1_active_sound': self.tone1_on > 0.5, # If sound is playing
            'tone2_active_sound': self.tone2_on > 0.5
        }

        # Apply move penalty if not staying
        if action != self.STAY:
            reward += self.move_penalty

        # Calculate next position
        dx, dy = self.ACTION_MAP[action]
        next_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        next_pos[0] = np.clip(next_pos[0], 0, self.width - 1)
        next_pos[1] = np.clip(next_pos[1], 0, self.height - 1)

        # Update agent position and zone
        self.agent_pos = next_pos
        new_zone = self._get_zone(self.agent_pos)
        crossed_midline = self.current_zone[0] != new_zone[0]

        # --- Tone Activation ---
        if not self.is_tone_active and not self.is_shock_active and self.steps_in_episode == self.tone_start_step:
            self.is_tone_active = True
            self.time_since_tone_onset = 0
            if self.active_tone_this_trial == 1:
                self.tone1_on = 1.0
            else: # active_tone_this_trial == 2
                self.tone2_on = 1.0
            # Store the zone where the agent was when the tone started
            self.zone_at_tone_onset = self.current_zone
            # Reset shuttle flag at tone onset
            self.shuttled_this_trial = False

        # Determine if the active tone is relevant for the current task
        is_relevant_tone_active_now = (
            (self.current_task_id == self.AVOID_TONE_1 and self.active_tone_this_trial == 1) or
            (self.current_task_id == self.AVOID_TONE_2 and self.active_tone_this_trial == 2)
        )
        # Update info dict
        if self.is_tone_active or self.is_shock_active or self.time_since_tone_onset > 0:
             info['is_relevant_tone'] = is_relevant_tone_active_now

        # Set shuttled flag if crossed midline *after* tone onset
        if (self.is_tone_active or self.is_shock_active or self.time_since_tone_onset > 0) and crossed_midline:
            self.shuttled_this_trial = True
            info['shuttled'] = True # Update info for this step

        # --- Tone Phase Logic ---
        if self.is_tone_active:
            self.time_since_tone_onset += 1

            # Check for Avoidance (Success)
            if is_relevant_tone_active_now and crossed_midline:
                # Successful avoidance of the relevant tone
                reward += self.avoidance_reward # possible positive reward for avoiding the shock
                # done = True # TODO: check if works better
                info['avoided'] = True
                self.is_tone_active = False # Stop tone logic TODO: check if needed
                # self.tone1_on = 0.0 # Turn off sound
                # self.tone2_on = 0.0

            # Check if tone duration expires
            elif self.time_since_tone_onset >= self.max_tone_duration_steps:
                self.is_tone_active = False # Stop tone logic
                self.tone1_on = 0.0 # Turn off sound
                self.tone2_on = 0.0
                # Shock check will happen in the next block if relevant

        # --- Post-Tone / Shock Phase Logic ---
        # Check if shock should start (only if tone phase ended naturally, not by avoidance)
        if not self.is_tone_active and not self.is_shock_active and not info['avoided'] and self.time_since_tone_onset > 0:
             # Check if the relevant tone was presented and delay is reached
             if is_relevant_tone_active_now and self.time_since_tone_onset >= self.shock_onset_delay_steps:
                 # Check if agent is still on the "wrong" side (where tone started)
                 if self.current_zone[0] == self.zone_at_tone_onset[0]:
                     self.is_shock_active = True # Start shock

        # --- Shock Active Logic ---
        if self.is_shock_active:
            reward += self.shock_penalty_per_step # Apply continuous shock penalty
            info['shocked'] = True # Mark that shock occurred this trial

            # Check for Escape
            if crossed_midline:
                # done = True # TODO: check if works better
                info['escaped'] = True
                self.is_shock_active = False # Stop shock
                self.is_tone_active = False

        # --- Update current zone for next step's comparison ---
        self.current_zone = new_zone

        # --- Check for Max Steps ---
        self.steps_in_episode += 1
        if self.steps_in_episode >= self.max_steps_per_episode:
            done = True
            # No specific timeout penalty, natural accumulation of step/shock penalties applies

        # Get the next state observation
        next_state = self._get_state()

        # Ensure tone flags are off if trial ended
        if done:
            self.tone1_on = 0.0
            self.tone2_on = 0.0
            self.is_tone_active = False
            self.is_shock_active = False
            # Final update to info dict based on trial end state
            info['shuttled'] = self.shuttled_this_trial
            if info['shocked'] and not info['escaped']:
                 # If shock occurred but wasn't escaped, it's a failure type
                 pass # Already captured by 'shocked'=True, 'escaped'=False


        return next_state, reward, done, info

    def render(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 2.5)) # Adjusted aspect ratio
            show = True
        else:
            show = False
            ax.clear()

        ax.scatter(self.agent_pos[0], self.agent_pos[1], marker='o', s=100, color='blue', label='Agent')
        ax.axvline(self.center_x, color='gray', linestyle='--', lw=1)

        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal', adjustable='box')

        tone1_str = "ON" if self.tone1_on > 0.5 else "OFF"
        tone2_str = "ON" if self.tone2_on > 0.5 else "OFF"
        time_str = f"{self.time_since_tone_onset}/{self.max_tone_duration_steps}" if self.is_tone_active or self.time_since_tone_onset > 0 else "--"
        shock_str = "ACTIVE" if self.is_shock_active else "off"
        task_str = f"Task: {'T1' if self.current_task_id == 1 else 'T2'}-Shuttle"
        title = (f"Step: {self.steps_in_episode}/{self.max_steps_per_episode} | {task_str} | T1:{tone1_str} T2:{tone2_str} ({time_str}) | Shock: {shock_str}")
        ax.set_title(title, fontsize=8)
        # ax.legend(loc='upper right') # Legend can clutter

        if show:
            plt.show()
