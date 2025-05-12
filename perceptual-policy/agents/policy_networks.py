import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MLPPolicyNetwork(nn.Module):
    """
    Flexible MLP Policy Network for discrete action spaces with configurable hidden layers
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(MLPPolicyNetwork, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        input_dim = state_dim
        
        # Add hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        # Create sequential model from layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def get_action(self, state):
        """
        Sample an action from the policy distribution
        Returns action, log probability, and entropy
        """
        logits = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

class RNNMLPPolicyNetwork(nn.Module):
    """
    RNN-based Policy Network with configurable hidden layers and size
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=1, rnn_type='gru'):
        super(RNNMLPPolicyNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Choose RNN type
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                state_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                batch_first=True
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                state_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                state_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                batch_first=True, 
                nonlinearity='relu'
            )
            
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x, hidden_state=None):
        # Handle single state vs sequence input
        if len(x.shape) == 1:  # Single state
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        elif len(x.shape) == 2 and x.shape[0] == 1:  # Batch of 1 state
            x = x.unsqueeze(1)  # Add sequence dim
        
        # Process with RNN
        if self.rnn_type == 'lstm':
            if hidden_state is None:
                out, (h, c) = self.rnn(x)
                hidden_output = (h, c)
            else:
                h, c = hidden_state
                out, (h, c) = self.rnn(x, (h, c))
                hidden_output = (h, c)
        else:
            out, h = self.rnn(x, hidden_state)
            hidden_output = h
            
        # Get action logits from the last output
        logits = self.fc(out[:, -1, :])
        
        return logits, hidden_output
    
    def get_action(self, state, hidden_state=None):
        """
        Sample an action from the policy distribution
        Returns action, log probability, entropy, and new hidden state
        """
        logits, h = self(state, hidden_state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, h
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state based on RNN type"""
        if self.rnn_type == 'lstm':
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            return (h, c)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)

class TransformerMLPPolicyNetwork(nn.Module):
    """
    Transformer-based Policy Network with configurable parameters
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_layers=2, nhead=4, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Linear(state_dim, hidden_dim)
        
        # Create transformer encoder layers with configurable parameters
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action head
        self.head = nn.Linear(hidden_dim, action_dim)
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # Handle single state vs sequence input
        if len(x.shape) == 1:  # Single state
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        elif len(x.shape) == 2 and x.shape[0] == 1:  # Batch of 1 state
            x = x.unsqueeze(1)  # Add sequence dim
            
        # x shape should now be (batch, seq_len, state_dim)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for transformer: (seq_len, batch, state_dim)
        x = x.transpose(0, 1)
        
        # Embed states
        x = self.embedding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Get final token and project to action space
        # Take the last token from sequence dimension
        x = x[-1]  # Shape: (batch, hidden_dim)
        
        return self.head(x)
    
    def get_action(self, state):
        """
        Sample an action from the policy distribution
        Returns action, log probability, and entropy
        """
        logits = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

def create_policy_network(policy_type, state_dim, action_dim, **kwargs):
    """
    Factory function to create a policy network of the specified type
    
    Args:
        policy_type: Type of policy network ("mlp", "rnn", or "transformer")
        state_dim: Input state dimension
        action_dim: Output action dimension
        **kwargs: Additional parameters for specific network types
            - For MLP: hidden_dims (list of int)
            - For RNN: hidden_dim (int), num_layers (int), rnn_type ('gru', 'lstm', 'rnn')
            - For Transformer: hidden_dim (int), num_layers (int), nhead (int),
                              dim_feedforward (int), dropout (float)
    
    Returns:
        A policy network instance
    """
    if policy_type == "mlp":
        hidden_dims = kwargs.get('hidden_dims', [64, 64])
        return MLPPolicyNetwork(state_dim, action_dim, hidden_dims)
        
    elif policy_type == "rnn":
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 1)
        rnn_type = kwargs.get('rnn_type', 'gru')
        return RNNMLPPolicyNetwork(state_dim, action_dim, hidden_dim, num_layers, rnn_type)
        
    elif policy_type == "transformer":
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 2)
        nhead = kwargs.get('nhead', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        dropout = kwargs.get('dropout', 0.1)
        return TransformerMLPPolicyNetwork(
            state_dim, action_dim, hidden_dim, num_layers, nhead, dim_feedforward, dropout
        )
        
    else:
        raise ValueError(f"Unknown policy network type: {policy_type}")