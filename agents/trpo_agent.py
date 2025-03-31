import torch
import numpy as np
from agents.base_agent import BaseAgent
from agents.policy_networks import create_policy_network

class TRPOAgent(BaseAgent):
    """
    Trust Region Policy Optimization Agent
    
    Implements the TRPO algorithm, which uses natural gradient and a trust region
    constraint to ensure stable policy updates.
    """
    
    def __init__(self, policy_type, state_dim, action_dim, hidden_dim=64, 
                 gamma=0.99, kl_delta=0.01, cg_iters=10, cg_damping=1e-2, **kwargs):
        """
        Initialize the TRPO agent
        
        Args:
            policy_type: Type of policy network ("mlp", "rnn", or "transformer")
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension size for policy network
            gamma: Discount factor
            kl_delta: KL constraint limit
            cg_iters: Number of conjugate gradient iterations
            cg_damping: Damping coefficient for conjugate gradient
        """
        # For TRPO, we don't use a learning rate with the optimizer
        # because we compute the step size adaptively
        super().__init__(policy_type, state_dim, action_dim, hidden_dim, lr=0.0, gamma=gamma, **kwargs)
        self.kl_delta = kl_delta
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
    
    def flat_concat(self, tensors):
        """Flatten and concatenate a list of tensors"""
        return torch.cat([t.contiguous().view(-1) for t in tensors])
    
    def get_flat_params(self):
        """Get policy parameters as a flattened vector"""
        return torch.cat([p.data.view(-1) for p in self.policy_net.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set parameters from a flat vector"""
        pointer = 0
        for p in self.policy_net.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer+numel].view_as(p))
            pointer += numel
    
    def surrogate_loss(self, states, actions, advantages, old_log_probs, hidden_states=None):
        """
        Compute the surrogate loss for TRPO
        
        Args:
            states: List of states from the trajectory
            actions: List of actions taken
            advantages: Advantage estimates
            old_log_probs: Log probabilities of the actions under the old policy
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            loss: Surrogate loss
            log_probs: Log probabilities of the actions under the current policy
        """
        if self.policy_type == "rnn":
            log_probs = []
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                hidden_state = hidden_states[i] if hidden_states and i < len(hidden_states) else None
                
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
        
        # Importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)
        loss = -torch.mean(ratio * advantages)
        return loss, log_probs
    
    def kl_divergence(self, states, old_logits, hidden_states=None):
        """
        Compute the KL divergence between old and current policy
        
        Args:
            states: List of states from the trajectory
            old_logits: Logits from the old policy
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            KL divergence
        """
        if self.policy_type == "rnn":
            kl_sum = 0
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                hidden_state = hidden_states[i] if hidden_states and i < len(hidden_states) else None
                
                logits, _ = self.policy_net(state, hidden_state)
                new_dist = torch.distributions.Categorical(logits=logits)
                old_dist = torch.distributions.Categorical(logits=old_logits[i])
                
                kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
                kl_sum += kl.mean()
            
            return kl_sum / len(states)
        else:
            # Standard KL for MLP and Transformer
            logits = self.policy_net(states)
            new_dist = torch.distributions.Categorical(logits=logits)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            
            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
            return torch.mean(kl)
    
    def fisher_vector_product(self, states, vector, old_logits=None, hidden_states=None):
        """
        Compute the Fisher-vector product for TRPO
        
        Args:
            states: List of states from the trajectory
            vector: Vector to compute product with
            old_logits: Logits from the old policy
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            Fisher-vector product
        """
        if self.policy_type == "rnn":
            kl_sum = 0
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                hidden_state = hidden_states[i] if hidden_states and i < len(hidden_states) else None
                
                logits, _ = self.policy_net(state, hidden_state)
                dist = torch.distributions.Categorical(logits=logits)
                
                with torch.no_grad():
                    old_logit = old_logits[i] if old_logits else logits.detach()
                    old_dist = torch.distributions.Categorical(logits=old_logit)
                
                kl = torch.distributions.kl.kl_divergence(old_dist, dist)
                kl_sum += kl.mean()
            
            kl = kl_sum / len(states)
        else:
            # Standard KL for MLP and Transformer
            logits = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            
            with torch.no_grad():
                old_dist = torch.distributions.Categorical(logits=old_logits if old_logits is not None else logits.detach())
            
            kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
        
        # Compute gradient of KL w.r.t. policy parameters
        grads = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
        flat_grad_kl = self.flat_concat(grads)
        
        # Compute Hessian-vector product
        grad_kl_v = torch.sum(flat_grad_kl * vector)
        hvp = torch.autograd.grad(grad_kl_v, self.policy_net.parameters(), retain_graph=True)
        flat_hvp = self.flat_concat(hvp)
        
        return flat_hvp + self.cg_damping * vector
    
    def conjugate_gradient(self, fvp_fn, b, max_iter, tol=1e-10, states=None, old_logits=None, hidden_states=None):
        """
        Conjugate gradient algorithm to solve Ax = b
        
        Args:
            fvp_fn: Function to compute Fisher-vector product
            b: Right-hand side of the equation
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            states: List of states from the trajectory
            old_logits: Logits from the old policy
            hidden_states: List of hidden states (for RNN policies)
            
        Returns:
            Approximate solution to Ax = b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            if self.policy_type == "rnn":
                Avp = fvp_fn(states, p, old_logits, hidden_states)
            else:
                Avp = fvp_fn(states, p, old_logits)
                
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def update(self, states, actions, rewards, log_probs_old, hidden_states=None):
        """
        Update policy using Trust Region Policy Optimization
        
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
        
        # Compute returns and use as advantages
        returns = self.compute_returns(rewards)
        advantages = returns
        
        # Get current policy logits to use for KL divergence computation
        if self.policy_type == "rnn":
            old_logits = []
            for i, state in enumerate(states):
                if isinstance(state, torch.Tensor) and len(state.shape) == 1:
                    state = state.unsqueeze(0)
                
                hidden_state = hidden_states[i] if i < len(hidden_states) else None
                
                with torch.no_grad():
                    logits, _ = self.policy_net(state, hidden_state)
                    old_logits.append(logits.detach())
        else:
            with torch.no_grad():
                old_logits = self.policy_net(states).detach()
        
        # Compute surrogate loss
        self.policy_net.zero_grad()
        loss, new_log_probs = self.surrogate_loss(states, actions, advantages, log_probs_old, hidden_states)
        
        # Get gradient of surrogate loss
        loss.backward(retain_graph=True)
        loss_grad = self.flat_concat([p.grad for p in self.policy_net.parameters()])
        self.policy_net.zero_grad()
        
        # Compute step direction using conjugate gradient
        if self.policy_type == "rnn":
            step_dir = self.conjugate_gradient(
                self.fisher_vector_product, loss_grad, self.cg_iters,
                states=states, old_logits=old_logits, hidden_states=hidden_states
            )
        else:
            step_dir = self.conjugate_gradient(
                self.fisher_vector_product, loss_grad, self.cg_iters,
                states=states, old_logits=old_logits
            )
        
        # Compute step size to satisfy KL constraint
        if self.policy_type == "rnn":
            shs = 0.5 * torch.sum(step_dir * self.fisher_vector_product(states, step_dir, old_logits, hidden_states))
        else:
            shs = 0.5 * torch.sum(step_dir * self.fisher_vector_product(states, step_dir, old_logits))
            
        lm = torch.sqrt(self.kl_delta / (shs + 1e-8))
        full_step = lm * step_dir
        
        # Line search to enforce KL constraint
        old_params = self.get_flat_params()
        expected_improve = -torch.sum(full_step * loss_grad)
        
        # Perform backtracking line search
        success = False
        backtrack_coeff = 1.0
        max_backtracks = 10
        
        for _ in range(max_backtracks):
            step = backtrack_coeff * full_step
            new_params = old_params - step
            self.set_flat_params(new_params)
            
            # Calculate new loss and KL
            new_loss, _ = self.surrogate_loss(states, actions, advantages, log_probs_old, hidden_states)
            
            if self.policy_type == "rnn":
                kl = self.kl_divergence(states, old_logits, hidden_states)
            else:
                kl = self.kl_divergence(states, old_logits)
            
            improvement = loss.item() - new_loss.item()
            
            if improvement > 0 and kl < self.kl_delta:
                success = True
                break
                
            backtrack_coeff *= 0.5
        
        if not success:
            # If line search failed, revert to old parameters
            self.set_flat_params(old_params)
            kl = torch.tensor(0.0)
        
        return {'loss': loss.item(), 'kl': kl.item()}