import torch
import numpy as np
from agents.base_agent import BaseAgent
from agents.policy_networks import create_policy_network

# Import gradient clipping utility
from torch.nn.utils import clip_grad_norm_

class MaxEntAgent(BaseAgent):
    """
    Maximum Entropy Reinforcement Learning Agent
    """
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64,
                 lr=0.01, temperature=0.1, gamma=0.99,
                 gradient_clip_norm=1.0, **kwargs):
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr, gamma, **kwargs)
        # Store initial temperature, but allow it to be changed
        self.temperature = temperature
        self.gradient_clip_norm = gradient_clip_norm

    # --- Add this method ---
    def set_temperature(self, new_temperature):
        """Allows updating the temperature parameter externally."""
        self.temperature = new_temperature
    # -----------------------

    def update(self, states, actions, rewards, log_probs_old, hidden_states=None):
        # ... (rest of the update method remains the same) ...
        # It will use the current value of self.temperature

        # Convert lists to tensors
        if isinstance(states[0], (list, np.ndarray)):
            states = torch.FloatTensor(np.array(states)).to(self.device)
        elif not isinstance(states, torch.Tensor):
             states = torch.FloatTensor(states).to(self.device)
        else:
             states = states.float().to(self.device) # Ensure float type

        actions = torch.LongTensor(actions).to(self.device)
        returns = self.compute_returns(rewards)

        total_policy_loss = 0
        total_pg_loss_component = 0
        total_entropy_loss_component = 0
        total_entropy = 0
        num_steps = len(states)

        if self.policy_type == "rnn":
            for i in range(num_steps):
                state = states[i].unsqueeze(0)
                action = actions[i].unsqueeze(0)
                G = returns[i]
                hidden_state = None
                if hidden_states and i < len(hidden_states):
                     if isinstance(hidden_states[i], tuple):
                         hidden_state = tuple(h.to(self.device) for h in hidden_states[i])
                     else:
                         hidden_state = hidden_states[i].to(self.device)

                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                step_pg_loss = -(log_prob * G)
                # Use the current self.temperature here
                step_entropy_loss = -(self.temperature * entropy)
                step_total_loss = step_pg_loss + step_entropy_loss

                total_policy_loss += step_total_loss
                total_pg_loss_component += step_pg_loss
                total_entropy_loss_component += step_entropy_loss
                total_entropy += entropy.item()

            avg_policy_loss = total_policy_loss / num_steps
            avg_pg_loss = total_pg_loss_component / num_steps
            avg_entropy_loss = total_entropy_loss_component / num_steps
            avg_entropy = total_entropy / num_steps

        else: # MLP / Transformer
            if len(states.shape) == 1: states = states.unsqueeze(0)
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            pg_loss_terms = -(log_probs * returns)
             # Use the current self.temperature here
            entropy_loss_terms = -(self.temperature * entropies)
            total_loss_terms = pg_loss_terms + entropy_loss_terms

            avg_policy_loss = torch.mean(total_loss_terms)
            avg_pg_loss = torch.mean(pg_loss_terms)
            avg_entropy_loss = torch.mean(entropy_loss_terms)
            avg_entropy = entropies.mean().item()

        self.optimizer.zero_grad()
        avg_policy_loss.backward()

        if self.gradient_clip_norm is not None:
            clip_grad_norm_(self.policy_net.parameters(), max_norm=self.gradient_clip_norm)

        self.optimizer.step()

        return {
            'policy_loss': avg_policy_loss.item(),
            'pg_loss': avg_pg_loss.item(),
            'entropy_loss': avg_entropy_loss.item(),
            'entropy': avg_entropy
        }
    """
    Maximum Entropy Reinforcement Learning Agent
    
    Implements the maximum entropy reinforcement learning algorithm, which 
    encourages exploration by maximizing policy entropy along with expected return.
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, 
                 lr=0.01, temperature=0.1, gamma=0.99, 
                 gradient_clip_norm=1.0, **kwargs): # Added gradient_clip_norm
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
            gradient_clip_norm: Value for gradient clipping (set to None to disable)
        """
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr, gamma, **kwargs)
        self.temperature = temperature
        self.gradient_clip_norm = gradient_clip_norm # Store clipping value
    
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
            Dictionary with update statistics including loss components
        """
        # Convert lists to tensors
        # Ensure states are float tensors
        if isinstance(states[0], (list, np.ndarray)):
            states = torch.FloatTensor(np.array(states)).to(self.device)
        elif not isinstance(states, torch.Tensor):
             states = torch.FloatTensor(states).to(self.device)
        else:
             states = states.float().to(self.device) # Ensure float type

        actions = torch.LongTensor(actions).to(self.device)
        # Use compute_returns which includes normalization
        returns = self.compute_returns(rewards) 
        
        # Initialize accumulators for loss components
        total_policy_loss = 0
        total_pg_loss_component = 0
        total_entropy_loss_component = 0
        total_entropy = 0
        num_steps = len(states)

        # Recompute log probabilities and entropies for gradient flow
        if self.policy_type == "rnn":
            for i in range(num_steps):
                state = states[i].unsqueeze(0) # Add batch dim for RNN
                action = actions[i].unsqueeze(0) # Add batch dim
                G = returns[i] # Return for this step
                # Ensure hidden_state is correctly formatted and on device
                hidden_state = None
                if hidden_states and i < len(hidden_states):
                     # Assuming hidden_states[i] is already a tuple (h, c) or single tensor h
                     # Ensure it's on the correct device
                     if isinstance(hidden_states[i], tuple):
                         hidden_state = tuple(h.to(self.device) for h in hidden_states[i])
                     else:
                         hidden_state = hidden_states[i].to(self.device)

                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                # Calculate components for this step
                step_pg_loss = -(log_prob * G)
                step_entropy_loss = -(self.temperature * entropy)
                step_total_loss = step_pg_loss + step_entropy_loss

                # Accumulate
                total_policy_loss += step_total_loss
                total_pg_loss_component += step_pg_loss
                total_entropy_loss_component += step_entropy_loss
                total_entropy += entropy.item() # Accumulate raw entropy

            # Average over sequence length
            avg_policy_loss = total_policy_loss / num_steps
            avg_pg_loss = total_pg_loss_component / num_steps
            avg_entropy_loss = total_entropy_loss_component / num_steps
            avg_entropy = total_entropy / num_steps

        else: # MLP / Transformer
            # Ensure states has batch dimension if needed by network
            if len(states.shape) == 1:
                 states = states.unsqueeze(0) # Add batch dim if single state passed

            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            # Calculate components over the batch
            pg_loss_terms = -(log_probs * returns)
            entropy_loss_terms = -(self.temperature * entropies)
            total_loss_terms = pg_loss_terms + entropy_loss_terms

            # Average over batch size
            avg_policy_loss = torch.mean(total_loss_terms)
            avg_pg_loss = torch.mean(pg_loss_terms)
            avg_entropy_loss = torch.mean(entropy_loss_terms)
            avg_entropy = entropies.mean().item() # Average raw entropy

        # Optimize
        self.optimizer.zero_grad()
        # Use the calculated average total loss for backpropagation
        avg_policy_loss.backward() 

        # Apply gradient clipping (if enabled)
        if self.gradient_clip_norm is not None:
            clip_grad_norm_(self.policy_net.parameters(), max_norm=self.gradient_clip_norm)

        self.optimizer.step()
        
        return {
            'policy_loss': avg_policy_loss.item(),
            'pg_loss': avg_pg_loss.item(),           # Avg Policy Gradient Loss Component
            'entropy_loss': avg_entropy_loss.item(), # Avg Entropy Bonus Loss Component
            'entropy': avg_entropy                   # Avg Raw Entropy
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
            #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
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