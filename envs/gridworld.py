import numpy as np

class GridWorld:
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    
    def __init__(self, grid_size=10, start=(0, 0), goal=(9, 9), max_steps=100, stochastic=False, noise=0.1):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.noise = noise
        self.obstacles = []
        self.reset()
        
    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()
        
    def _get_state(self):
        return np.array(self.agent_pos, dtype=np.float32) / (self.grid_size - 1)
        
    def step(self, action):
        if self.stochastic and np.random.random() < self.noise:
            action = np.random.randint(0, 4)
            
        next_pos = list(self.agent_pos)
        if action == self.UP and self.agent_pos[1] < self.grid_size - 1:
            next_pos[1] += 1
        elif action == self.DOWN and self.agent_pos[1] > 0:
            next_pos[1] -= 1
        elif action == self.LEFT and self.agent_pos[0] > 0:
            next_pos[0] -= 1
        elif action == self.RIGHT and self.agent_pos[0] < self.grid_size - 1:
            next_pos[0] += 1
            
        if tuple(next_pos) not in self.obstacles:
            self.agent_pos = next_pos

        self.steps += 1
        state = self._get_state()
        
        done = False
        reward = -1
        
        if tuple(self.agent_pos) == self.goal:
            reward = 50.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
            
        return state, reward, done, {}