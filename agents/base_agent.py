import numpy as np
import torch
import torch.optim as optim
from policy_networks import create_policy_network

class BaseAgent:
    """
    Base class for all reinforcement learning agents with common functionality
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, lr=0.01, gamma=0.99, **kwargs):
        """
        Initialize the base agent
        
        Args:
            policy_type: Type of policy network ("mlp", "rnn", or "transformer")
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension size for policy network
            lr: Learning rate
            gamma: Discount factor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the policy network
        self.policy_net = create_policy_network(
            policy_type, state_dim, action_dim, hidden_dim, **kwargs
        ).to(self.device)
        
        self.policy_type = policy_type
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        
    def select_action(self, state, hidden_state=None):
        """
        Select an action using the current policy
        
        Args:
            state: Current environment state
            hidden_state: Hidden state for RNN policies (if applicable)
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            entropy: Entropy of the action distribution
            hidden_state: New hidden state (for RNN policies only)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            if self.policy_type == "rnn":
                action, log_prob, entropy, h = self.policy_net.get_action(state_tensor, hidden_state)
                return action.item(), log_prob.item(), entropy.item(), h
            else:
                action, log_prob, entropy = self.policy_net.get_action(state_tensor)
                return action.item(), log_prob.item(), entropy.item()
    
    def compute_returns(self, rewards):
        """
        Compute discounted returns from rewards
        
        Args:
            rewards: List of rewards for each timestep
            
        Returns:
            returns: Discounted returns for each timestep
        """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stable learning
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, *args, **kwargs):
        """
        Update the policy (to be implemented by subclasses)
        """
        raise NotImplementedError
    
    def train(self, env, num_episodes=500, max_steps=1000):
        """
        Train the agent on an environment
        
        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            
        Returns:
            rewards_history: List of rewards per episode
            losses_history: List of losses per update
            metric_history: List of additional metrics per update
        """
        rewards_history = []
        losses_history = []
        metric_history = []  # For additional metrics (e.g., entropy, KL)
        
        for episode in range(num_episodes):
            # Initialize environment and episode buffers
            state = env.reset()
            states = []
            actions = []
            rewards = []
            log_probs = []
            hidden_states = []
            episode_reward = 0
            
            # Initialize hidden state for RNN policies
            hidden_state = None
            if self.policy_type == "rnn":
                hidden_state = self.policy_net.init_hidden()
            
            # Collect trajectory
            for step in range(max_steps):
                # Select action
                if self.policy_type == "rnn":
                    action, log_prob, _, h = self.select_action(state, hidden_state)
                    hidden_states.append(hidden_state)
                    hidden_state = h
                else:
                    action, log_prob, _ = self.select_action(state)
                
                # Take action in environment
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                # Update tracking variables
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update policy based on collected trajectory
            if self.policy_type == "rnn":
                update_info = self.update(states, actions, rewards, log_probs, hidden_states)
            else:
                update_info = self.update(states, actions, rewards, log_probs)
            
            # Record history
            rewards_history.append(episode_reward)
            losses_history.append(update_info.get('loss', update_info.get('policy_loss', 0)))
            
            # Get the appropriate metric based on the agent type
            if 'entropy' in update_info:
                metric_history.append(update_info['entropy'])
            elif 'approx_kl' in update_info:
                metric_history.append(update_info['approx_kl'])
            elif 'kl' in update_info:
                metric_history.append(update_info['kl'])
            else:
                metric_history.append(0)  # Fallback
            
            # Print progress
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(rewards_history[-20:])
                avg_loss = np.mean(losses_history[-20:])
                avg_metric = np.mean(metric_history[-20:])
                
                metric_name = 'Entropy' if 'entropy' in update_info else ('KL' if 'kl' in update_info or 'approx_kl' in update_info else 'Metric')
                print(f"Episode {episode+1}/{num_episodes}, Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}, {metric_name}: {avg_metric:.4f}")
        
        return rewards_history, losses_history, metric_history