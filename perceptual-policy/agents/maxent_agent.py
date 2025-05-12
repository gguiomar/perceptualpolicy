import torch
import numpy as np
from agents.base_agent import BaseAgent
from agents.policy_networks import create_policy_network

class MaxEntAgent(BaseAgent):
    """
    Maximum Entropy Reinforcement Learning Agent
    
    Implements the maximum entropy reinforcement learning algorithm, which 
    encourages exploration by maximizing policy entropy along with expected return.
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, 
                 lr=0.01, temperature=0.1, gamma=0.99, **kwargs):
        """
        Initialize the MaxEnt agent
        
        Args:
            policy_type: Type of policy network ("mlp", "rnn", or "transformer")
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension size for policy network
            lr: Learning rate
            temperature: Temperature parameter for entropy regularization
            gamma: Discount factor
        """
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr, gamma, **kwargs)
        self.temperature = temperature
    
    def update(self, states, actions, rewards, log_probs_old, hidden_states=None):
        """
        Update the policy using maximum entropy reinforcement learning
        
        Args:
            states: List of states from the trajectory
            actions: List of actions taken
            rewards: List of rewards received
            log_probs_old: Log probabilities of the actions (unused, included for API consistency)
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
        returns = self.compute_returns(rewards)
        
        # Recompute log probabilities and entropies for gradient flow
        if self.policy_type == "rnn":
            # Create sequences for RNN processing
            policy_loss = 0
            entropy_sum = 0
            
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                G = returns[i]
                hidden_state = hidden_states[i] if i < len(hidden_states) else None
                
                # Get logits from the policy network
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Add to policy loss with entropy bonus
                policy_loss += -(log_prob * G + self.temperature * entropy)
                entropy_sum += entropy.item()
            
            policy_loss = policy_loss / len(states)
            avg_entropy = entropy_sum / len(states)
        else:
            # Standard forward pass for MLP and Transformer
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            
            # Compute policy loss with entropy bonus
            policy_loss = -torch.mean(log_probs * returns + self.temperature * entropies)
            avg_entropy = entropies.mean().item()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'entropy': avg_entropy
        }


class FisherMaxEntAgent(MaxEntAgent):
    """
    Maximum Entropy Reinforcement Learning Agent with Natural Gradient
    
    Extends the MaxEnt agent with natural gradient updates using the
    Fisher Information Matrix for more efficient policy updates.
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, 
                 lr=0.01, temperature=0.1, gamma=0.99,
                 use_natural_gradient=True, cg_iters=10, cg_damping=1e-2, **kwargs):
        """
        Initialize the Fisher MaxEnt agent
        
        Args:
            policy_type: Type of policy network ("mlp", "rnn", or "transformer")
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension size for policy network
            lr: Learning rate
            temperature: Temperature parameter for entropy regularization
            gamma: Discount factor
            use_natural_gradient: Whether to use natural gradient updates
            cg_iters: Number of conjugate gradient iterations
            cg_damping: Damping coefficient for conjugate gradient
        """
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr, temperature, gamma, **kwargs)
        self.use_natural_gradient = use_natural_gradient
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
    
    def flat_concat(self, tensors):
        """Flatten and concatenate a list of tensors"""
        return torch.cat([t.view(-1) for t in tensors])
    
    def get_flat_params(self):
        """Get policy parameters as a single flattened vector"""
        return self.flat_concat([p.data for p in self.policy_net.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set parameters from a flat vector"""
        pointer = 0
        for p in self.policy_net.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
            pointer += numel
    
    def get_flat_grad(self):
        """Get policy gradients as a single flattened vector"""
        return self.flat_concat([p.grad.view(-1) for p in self.policy_net.parameters() if p.grad is not None])
    
    def fisher_vector_product(self, states, vector, hidden_states=None):
        """
        Compute F*v where F is the Fisher Information Matrix (FIM)
        
        Args:
            states: List of states from the trajectory
            vector: Vector to compute product with
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            Fisher-vector product
        """
        # Get current logits and distribution
        if self.policy_type == "rnn":
            # For RNN, we need to handle the sequence properly
            kl_sum = 0
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                hidden_state = hidden_states[i] if hidden_states and i < len(hidden_states) else None
                
                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                
                # Detached copy for "old" policy
                with torch.no_grad():
                    old_logits = logits.detach()
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                
                # Compute KL
                kl = torch.distributions.kl.kl_divergence(old_dist, dist)
                kl_sum += kl.mean()
            
            # Average KL
            kl = kl_sum / len(states)
        else:
            # Standard forward pass for MLP and Transformer
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            
            # Detached copy for "old" policy
            with torch.no_grad():
                old_logits = logits.detach()
                old_dist = torch.distributions.Categorical(logits=old_logits)
            
            # Compute KL
            kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        
        # Compute gradient of KL w.r.t. parameters
        grad_kl = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad_kl = self.flat_concat(grad_kl)
        
        # Compute the inner product
        grad_kl_dot_vector = torch.dot(flat_grad_kl, vector)
        hv = torch.autograd.grad(grad_kl_dot_vector, self.policy_net.parameters(), retain_graph=True)
        flat_hv = self.flat_concat(hv).detach()
        
        # Add damping for numerical stability
        return flat_hv + self.cg_damping * vector
    
    def conjugate_gradient(self, fvp_fn, b, nsteps, tol=1e-10, states=None, hidden_states=None):
        """
        Conjugate gradient method to solve Ax = b
        
        Args:
            fvp_fn: Function to compute Fisher-vector product
            b: Right-hand side of the equation
            nsteps: Maximum number of steps
            tol: Convergence tolerance
            states: List of states from the trajectory
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            Approximate solution to Ax = b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(nsteps):
            if self.policy_type == "rnn":
                Av = fvp_fn(states, p, hidden_states)
            else:
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
    
    def update(self, states, actions, rewards, log_probs_old, hidden_states=None):
        """
        Update policy using natural gradient (optional) and entropy regularization
        
        Args:
            states: List of states from the trajectory
            actions: List of actions taken
            rewards: List of rewards received
            log_probs_old: Log probabilities of the actions (unused, included for API consistency)
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
        returns = self.compute_returns(rewards)
        
        # Compute the max entropy policy loss
        if self.policy_type == "rnn":
            # For RNN, process each timestep in the sequence
            policy_loss = 0
            entropy_sum = 0
            
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                G = returns[i]
                hidden_state = hidden_states[i] if i < len(hidden_states) else None
                
                # Get logits from the policy network
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Add to policy loss with entropy bonus
                policy_loss += -(log_prob * G + self.temperature * entropy)
                entropy_sum += entropy.item()
            
            policy_loss = policy_loss / len(states)
            avg_entropy = entropy_sum / len(states)
        else:
            # Standard forward pass for MLP and Transformer
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
            
            # Compute policy loss with entropy bonus
            policy_loss = -torch.mean(log_probs * returns + self.temperature * entropies)
            avg_entropy = entropies.mean().item()
        
        if self.use_natural_gradient:
            # Natural gradient update
            self.optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            flat_grad = self.get_flat_grad()
            
            # Compute natural gradient direction
            if self.policy_type == "rnn":
                ng_direction = self.conjugate_gradient(
                    self.fisher_vector_product, flat_grad, self.cg_iters, 
                    states=states, hidden_states=hidden_states
                )
            else:
                ng_direction = self.conjugate_gradient(
                    self.fisher_vector_product, flat_grad, self.cg_iters, 
                    states=states
                )
            
            # Update parameters manually
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
            'entropy': avg_entropy
        }