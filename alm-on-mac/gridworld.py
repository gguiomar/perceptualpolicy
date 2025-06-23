import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Simple space definitions to replace gym.spaces
class DiscreteSpace:
    def __init__(self, n):
        self.n = n


    def sample(self):
        return random.randint(0, self.n - 1)
        
class BoxSpace:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

class GridWorld:
    """
    Simple GridWorld environment with configurable start and goal positions
    """
    # Action definitions
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    
    def __init__(self, grid_size=10, start=(0, 0), goal=(9,9), max_steps=100, 
                 stochastic=False, noise=0.1, add_obstacles=False, custom_obstacles=None):
        """
        Initialize GridWorld environment
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            start: Starting position (x, y)
            goal: Goal position (x, y)
            max_steps: Maximum steps before termination
            stochastic: Whether actions have random outcomes
            noise: Probability of random action being taken
            add_obstacles: Whether to add default obstacles
            custom_obstacles: List of custom obstacle positions, overrides default if provided
        """
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.noise = noise
        self.add_obstacles = add_obstacles
        self.success = False

        self.visit_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.final_goal = (9,9)
        
        # Define action and observation spaces for compatibility with agents
        self.action_space = DiscreteSpace(4)  # UP, DOWN, LEFT, RIGHT
        self.observation_space = BoxSpace(
            low=0, high=1, shape=(2,), dtype=np.float32
        )  # x, y normalized positions
        
        # Generate obstacles
        if custom_obstacles is not None:
            self.obstacles = custom_obstacles
        elif add_obstacles:
            self.obstacles = self._generate_obstacles()
        else:
            self.obstacles = []
        
        # Initialize agent position
        self.agent_pos = None
        self.steps = 0
        self.reset(train_episodes=0,eval=True)

        # Enable plotting
        plt.ion()
    
    def _generate_obstacles(self):
        """Generate obstacles in the grid"""
        obstacles = []
        
        # Add a vertical wall with a gap
        wall_x = self.grid_size // 2
        for y in range(self.grid_size):
            if y != self.grid_size // 3 and y != 2 * self.grid_size // 3:
                obstacles.append((wall_x, y))
        
        # Add a horizontal wall with a gap
        wall_y = self.grid_size // 2
        for x in range(self.grid_size):
            if x != self.grid_size // 4 and x != 3 * self.grid_size // 4:
                obstacles.append((x, wall_y))
        
        # Remove any obstacles that block start or goal
        obstacles = [obs for obs in obstacles 
                     if obs != self.start and obs != self.goal]
        
        return obstacles
    
    def reset(self, train_episodes, eval):
        """Reset the environment to initial state"""
        
        self.steps = 0
        #max_random_start_episodes = 100000
        #if not eval:
        #    if (train_episodes <= max_random_start_episodes):
        #        while True:
        #            self.agent_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        #            if self.agent_pos != list(self.goal):
        #                break
        #    else:
        #       self.agent_pos = list(self.start) 
        #elif eval:
        #    self.agent_pos = list(self.start)
        self.agent_pos = list(self.start)
        self.success = False
        return self._get_state()

    
    def _get_state(self):
        """Convert agent position to normalized state representation"""
        return np.array(self.agent_pos, dtype=np.float32) / (self.grid_size - 1)
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: Action to take (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
            
        Returns:
            state: New state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Apply random action if environment is stochastic
        if self.stochastic and np.random.random() < self.noise:
            action = np.random.randint(0, 4)
        
        # Compute distance before and after the move
        prev_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))

        # Move agent based on action
        next_pos = list(self.agent_pos)
        
        if action == self.UP and self.agent_pos[1] < self.grid_size - 1:
            next_pos[1] += 1
        elif action == self.DOWN and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == self.LEFT and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == self.RIGHT and self.agent_pos[0] < self.grid_size - 1:
            next_pos[0] += 1
        
        # Check if new position is valid (not an obstacle)
        if tuple(next_pos) not in self.obstacles:
            self.agent_pos = next_pos
            self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        
        self.steps += 1
        
        # Get new state
        state = self._get_state()

        # Compute new distance
        new_distance = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal))
        max_distance = np.linalg.norm(np.array(self.goal) - np.array(self.start))
        
        x, y = self.agent_pos

        # Reward logic
        if tuple(self.agent_pos) == self.goal:
            reward = 50.0
            done = True
            self.success = True

        else:

            # Gaussian reward centered at the final goal
            sigma = 6.0  # smoothness parameter
            dx = x - self.goal[0]
            dy = y - self.goal[1]
            dist_sq = dx ** 2 + dy ** 2
            reward = 50.0 * np.exp(-dist_sq / (2 * sigma ** 2))
            self.success = False
            done = False

        # Check if max steps reached
        if self.steps >= self.max_steps:
            done = True


        return state, reward, done, {"episode": {"r": reward, "l": self.steps}}, new_distance, prev_distance, self.success
    
    def render(self, mode='human'):
        visit_display = self.visit_counts.T  # Transpose so (0,0) is bottom-left
        plt.clf()
        plt.imshow(visit_display, origin='lower', cmap='viridis')
        plt.colorbar(label='Visit Count')
        plt.title(f"Agent Visit Heatmap - Step {step}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.pause(0.001)
    
    def plot_visit_heatmap(self, step):
        save_dir="visit_maps"
        os.makedirs(save_dir, exist_ok=True)
        visit_display = self.visit_counts.T  # Transpose so (0,0) is bottom-left
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.imshow(visit_display, origin='lower', cmap='viridis')
        plt.colorbar(label='Visit Count')
        plt.title(f"Agent Visit Heatmap - Step {step}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(os.path.join(save_dir, f"visit_map_step_{step}.png"))
        plt.close()

    
    def save_trajectory_plot(self, trajectory, episode_num, save_dir="trajectories"):
        os.makedirs(save_dir, exist_ok=True)

        xs, ys = zip(*trajectory)
        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, marker='o', linestyle='-', color='blue', label='Trajectory')
        plt.scatter(xs[0], ys[0], color='green', s=100, label='Start')
        plt.scatter(xs[-1], ys[-1], color='red', s=100, label='Goal')
        plt.title(f"Episode {episode_num} - Trajectory")
        plt.grid(True)
        plt.xlim(0, self.grid_size - 1)
        plt.ylim(0, self.grid_size - 1)
        plt.xticks(range(self.grid_size))
        plt.yticks(range(self.grid_size))
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(save_dir, f"trajectory_episode_{episode_num}.png"))
        plt.close()