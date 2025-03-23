import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random



class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, state):
        logits = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

class MaxEntAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.01, temperature=0.1, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.temperature = temperature
        self.gamma = gamma
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.policy_net.get_action(state_tensor)
        return action.item(), log_prob, entropy
    
    def update(self, states, actions, rewards):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # Recompute log probabilities and entropies (needed for gradient flow)
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        # Compute policy loss with entropy bonus
        policy_loss = 0
        for log_prob, G, entropy in zip(log_probs, returns, entropies):
            # Policy gradient with entropy regularization
            # This implements Equation (11) from the paper
            policy_loss += -(log_prob * G + self.temperature * entropy)
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item() / len(rewards),
            'entropy': entropies.mean().item()
        }

    def train(self, env, num_episodes=500, max_steps=100):
        rewards_history = []
        losses_history = []
        entropies_history = []
        
        for episode in range(num_episodes):
            state = env.reset()
            states = []
            actions = []
            rewards = []
            episode_reward = 0
            
            for step in range(max_steps):
                action, _, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            update_info = self.update(states, actions, rewards)
            rewards_history.append(episode_reward)
            losses_history.append(update_info['policy_loss'])
            entropies_history.append(update_info['entropy'])
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:])
                avg_loss = np.mean(losses_history[-20:])
                avg_entropy = np.mean(entropies_history[-20:])
                print(f"Episode {episode+1}/{num_episodes}, Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}")
        
        return rewards_history, losses_history, entropies_history

