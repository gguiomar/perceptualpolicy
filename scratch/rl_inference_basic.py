#%%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, grid_size=10, start=(0, 0), goal=(9, 9), max_steps=100):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return np.array(self.agent_pos, dtype=np.float32) / (self.grid_size - 1)

    def step(self, action):
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1

        self.steps += 1
        state = self._get_state()
        done = False
        reward = -0.1  
        if tuple(self.agent_pos) == self.goal:
            reward = 10.0  
            done = True
        if self.steps >= self.max_steps:
            done = True

        return state, reward, done, {}


class MLPPolicyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=4):
        super(MLPPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)  
        return logits

def select_action(policy_net, state):
    state_tensor = torch.from_numpy(state).unsqueeze(0)  
    logits = policy_net(state_tensor)
    probs = torch.softmax(logits, dim=1)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action), m.entropy()

def run_episode(env, policy_net):
    state = env.reset()
    log_probs = []
    rewards = []
    entropies = []
    positions = [env.agent_pos.copy()]
    done = False

    while not done:
        action, log_prob, entropy = select_action(policy_net, state)
        next_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        positions.append(env.agent_pos.copy())
        state = next_state

    return log_probs, rewards, entropies, positions

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns




def train_policy(num_episodes=500, alpha=0.01, gamma=0.99, lrate=0.01):
    env = GridWorld(grid_size=10, start=(0, 0), goal=(9, 9), max_steps=100)
    policy_net = MLPPolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=lrate)

    losses = []
    episode_rewards = []

    for i_episode in range(num_episodes):
        log_probs, rewards, entropies, _ = run_episode(env, policy_net)
        returns = compute_returns(rewards, gamma)

        loss = 0
        for log_prob, R, entropy in zip(log_probs, returns, entropies):
            loss += -log_prob * R - alpha * entropy
        loss = loss / len(rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        episode_rewards.append(sum(rewards))

        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode + 1}: Total Reward: {sum(rewards):.2f}, Loss: {loss.item():.2f}")

    return policy_net, losses, episode_rewards




def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Evolution of Loss During Training")
    plt.legend()
    plt.savefig("loss_evolution.png")
    plt.show()

def plot_trajectories(policy_net, num_samples=100):
    env = GridWorld(grid_size=10, start=(0, 0), goal=(9, 9), max_steps=100)
    plt.figure(figsize=(6, 6))
    for i in range(num_samples):
        _, rewards, _, positions = run_episode(env, policy_net)
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], marker='o', label=f"Trajectory {i+1}")
    plt.scatter([0], [0], c='green', s=100, label="Start")
    plt.scatter([9], [9], c='red', s=100, label="Goal")
    plt.xlim(-1, env.grid_size)
    plt.ylim(-1, env.grid_size)
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.title("Sample Trajectories After Training")
    plt.show()

def plot_policy_heatmaps(policy_net, grid_size=10):
    action_heatmaps = {0: np.zeros((grid_size, grid_size)),
                       1: np.zeros((grid_size, grid_size)),
                       2: np.zeros((grid_size, grid_size)),
                       3: np.zeros((grid_size, grid_size))}
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = np.array([i, j], dtype=np.float32) / (grid_size - 1)
            state_tensor = torch.from_numpy(state).unsqueeze(0)  
            logits = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=1).detach().numpy().squeeze()  
            for action in range(4):
                action_heatmaps[action][j, i] = probs[action]  

    actions = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    for action in range(4):
        im = axs[action].imshow(action_heatmaps[action], origin='lower', cmap='viridis', extent=[0, grid_size-1, 0, grid_size-1])
        axs[action].set_title(f"Action: {actions[action]}")
        axs[action].set_xlabel("X")
        axs[action].set_ylabel("Y")
        fig.colorbar(im, ax=axs[action])
    plt.suptitle("Policy Heatmaps: Probability of Taking Each Action per State")
    plt.savefig("policy_heatmaps.png")
    plt.show()




#%%
trained_policy, losses, ep_rewards = train_policy(num_episodes=500, alpha=0.1, gamma=0.99)
plot_loss(losses)
plot_trajectories(trained_policy, num_samples=10)
plot_policy_heatmaps(trained_policy, grid_size=10)
# %%
