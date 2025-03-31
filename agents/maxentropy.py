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



class FisherMaxEntAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.01, temperature=0.1, gamma=0.99,
                 use_natural_gradient=False, cg_iters=10, cg_damping=1e-2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.temperature = temperature
        self.gamma = gamma
        self.use_natural_gradient = use_natural_gradient
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.policy_net.get_action(state_tensor)
        return action.item(), log_prob, entropy

    def flat_concat(self, tensors):
        return torch.cat([t.view(-1) for t in tensors])
    
    def get_flat_params(self):
        return self.flat_concat([p.data for p in self.policy_net.parameters()])
    
    def set_flat_params(self, flat_params):
        # Set parameters from a flat vector
        pointer = 0
        for p in self.policy_net.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
            pointer += numel
    
    def get_flat_grad(self):
        return self.flat_concat([p.grad.view(-1) for p in self.policy_net.parameters() if p.grad is not None])
    
    def fisher_vector_product(self, states, vector):
        """
        Compute F*v where F is the Fisher Information Matrix (FIM) approximated via the KL divergence.
        """
        # Get current logits and distribution
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        
        # Use a detached copy as "old" policy
        with torch.no_grad():
            old_logits = logits.detach()
            old_dist = torch.distributions.Categorical(logits=old_logits)
        
        # Compute average KL divergence between old and current policy
        kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        
        # Compute gradient of KL w.r.t. parameters
        grad_kl = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad_kl = self.flat_concat(grad_kl)
        
        # Compute the inner product of grad_kl with the given vector
        grad_kl_dot_vector = torch.dot(flat_grad_kl, vector)
        hv = torch.autograd.grad(grad_kl_dot_vector, self.policy_net.parameters(), retain_graph=True)
        flat_hv = self.flat_concat(hv).detach()
        # Add damping for numerical stability
        return flat_hv + self.cg_damping * vector
    
    def conjugate_gradient(self, fvp_fn, b, nsteps, tol=1e-10, states=None):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Av = fvp_fn(states, p)
            alpha = rdotr / (torch.dot(p, Av) + 1e-8)
            x += alpha * p
            r -= alpha * Av
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

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
        
        # Recompute logits, distribution, log_probs, and entropies for gradient flow
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        # Compute the max entropy policy loss
        # Note: we minimize loss = - (log_prob * return + temperature * entropy)
        policy_loss = -torch.mean(log_probs * returns + self.temperature * entropies)
        
        if self.use_natural_gradient:
            # Natural gradient update:
            # 1. Zero existing gradients and compute gradient of loss
            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            flat_grad = self.get_flat_grad()
            
            # 2. Compute natural gradient direction using conjugate gradient
            ng_direction = self.conjugate_gradient(self.fisher_vector_product, flat_grad, self.cg_iters, states=states)
            
            # 3. Update parameters manually: theta_new = theta_old - lr * natural_gradient
            flat_params = self.get_flat_params()
            new_flat_params = flat_params - self.optimizer.param_groups[0]['lr'] * ng_direction
            self.set_flat_params(new_flat_params)
            
            # Zero gradients after manual update
            self.policy_net.zero_grad()
        else:
            # Standard update with Adam optimizer
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
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


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-2, epsilon=0.1, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.epsilon = epsilon  # clipping parameter: how far new policy can deviate
        self.gamma = gamma
        
    def select_action(self, state):
        """
        Given a state (as a numpy array), return:
         - chosen action (as an integer)
         - log probability of that action
         - entropy (for diagnostic purposes)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.policy_net.get_action(state_tensor)
        return action.item(), log_prob.item(), entropy.item()
    
    def update(self, states, actions, advantages, log_probs_old):
        """
        Perform a PPO update.
          states: list or np.array of states encountered during rollout.
          actions: list of actions (as integers) taken.
          advantages: computed advantage estimates for each timestep.
          log_probs_old: log probabilities of the actions under the old policy.
          
        This function implements the PPO clipped surrogate objective:
        
           L_CLIP = E[min(r(θ)*A, clip(r(θ),1-ε,1+ε)*A)]
           
        where r(θ) = exp(logπ(θ)(a|s) - logπ_old(a|s)).
        """
        # Convert to torch tensors and move to device.
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        
        # Recompute the current log probabilities for the rollout states.
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        
        # Compute the probability ratio r(θ) = exp(logπ - logπ_old)
        r_theta = torch.exp(log_probs - log_probs_old)
        # Clip the ratio within [1-ε, 1+ε]
        clipped_ratio = torch.clamp(r_theta, 1 - self.epsilon, 1 + self.epsilon)
        # PPO objective: take the elementwise minimum between the unclipped and clipped objective.
        surrogate_loss = torch.min(r_theta * advantages, clipped_ratio * advantages)
        loss = -torch.mean(surrogate_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Approximate KL divergence for diagnostics.
        approx_kl = torch.mean(log_probs_old - log_probs).item()
        return {'loss': loss.item(), 'approx_kl': approx_kl}
    
    def train(self, env, num_episodes=500, max_steps=100):
        """
        A simple training loop. In each episode, the agent collects a trajectory and
        then performs a PPO update. In practice, one might use mini-batch updates over
        a buffer of trajectories.
        """
        rewards_history = []
        losses_history = []
        kl_history = []
        
        for episode in range(num_episodes):
            state = env.reset()
            states, actions, rewards, log_probs_old = [], [], [], []
            episode_reward = 0
            
            for step in range(max_steps):
                action, log_prob, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs_old.append(log_prob)
                
                episode_reward += reward
                state = next_state
                if done:
                    break
            
            # Compute returns (discounted sum of rewards)
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = np.array(returns)
            # Normalize advantages (here we use returns as a simple proxy; one might subtract a value function)
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Perform PPO update. In many implementations, one performs multiple epochs over the collected batch.
            update_info = self.update(states, actions, advantages, log_probs_old)
            
            rewards_history.append(episode_reward)
            losses_history.append(update_info['loss'])
            kl_history.append(update_info['approx_kl'])
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:])
                avg_loss = np.mean(losses_history[-20:])
                avg_kl = np.mean(kl_history[-20:])
                print(f"Episode {episode+1}/{num_episodes}, Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}, Approx KL: {avg_kl:.6f}")
        
        return rewards_history, losses_history, kl_history