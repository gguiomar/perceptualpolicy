import torch
import numpy as np
from base_agent import BaseAgent

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization Agent
    
    Implements the PPO algorithm with clipped surrogate objective
    for stable policy updates.
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, 
                 lr=1e-2, epsilon=0.2, gamma=0.99, **kwargs):
        """
        Initialize the PPO agent
        
        Args:
            policy_type: Type of policy network ("mlp", "rnn", or "transformer")
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension size for policy network
            lr: Learning rate
            epsilon: Clipping parameter
            gamma: Discount factor
        """
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr, gamma, **kwargs)
        self.epsilon = epsilon  # Clipping parameter
    
    def update(self, states, actions, rewards, log_probs_old, hidden_states=None):
        """
        Perform a PPO update using clipped surrogate objective
        
        Args:
            states: List of states from the trajectory
            actions: List of actions taken
            rewards: List of rewards received
            log_probs_old: Log probabilities of the actions under the old policy
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            Dictionary with update statistics
        """
        # Convert lists to tensors
        if isinstance(states[0], list) or isinstance(states[0], np.ndarray):
            states = torch.FloatTensor(np.array(states)).to(self.device)
        else:
            states = torch.FloatTensor(states).to(self.device)
            
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        
        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        advantages = returns  # Using returns as advantages (simple version)
        
        # Recompute log probabilities for the current policy
        if self.policy_type == "rnn":
            log_probs = []
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                hidden_state = hidden_states[i] if i < len(hidden_states) else None
                
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs)
        else:
            # Standard forward pass for MLP and Transformer
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
        
        # Compute probability ratio r(θ) = exp(logπ - logπ_old)
        r_theta = torch.exp(log_probs - log_probs_old)
        
        # Clipped objective
        clipped_ratio = torch.clamp(r_theta, 1 - self.epsilon, 1 + self.epsilon)
        surrogate_loss = torch.min(r_theta * advantages, clipped_ratio * advantages)
        loss = -torch.mean(surrogate_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Approximate KL divergence for diagnostics
        approx_kl = torch.mean(log_probs_old - log_probs).item()
        
        return {'loss': loss.item(), 'approx_kl': approx_kl}