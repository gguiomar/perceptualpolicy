import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import utils
from utils import torch_utils

class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dims)
        )

    def forward(self, x):
        return self.encoder(x)  # [batch_size, latent_dim]
        
class ModelPrior(nn.Module):
    def __init__(self, latent_dims, num_actions, hidden_dims, num_layers=2):
        super().__init__()
        self.latent_dims = latent_dims
        self.std_min = 0.1
        self.std_max = 10.0
        self.action_embedding_dim = 4

        self.action_embedding = nn.Embedding(num_actions, self.action_embedding_dim)
        self.input_dim = latent_dims + self.action_embedding_dim

        self.model = self._build_model(hidden_dims, num_layers)
        self.apply(torch_utils.weight_init)

    def _build_model(self, hidden_dims, num_layers):
        layers = [nn.Linear(self.input_dim, hidden_dims), nn.ELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dims, hidden_dims), nn.ELU()]
        layers += [nn.Linear(hidden_dims, 2 * self.latent_dims)]
        return nn.Sequential(*layers)

    def forward(self, z, action):
        a_embed = self.action_embedding(action)
        x = torch.cat([z, a_embed], dim=-1)
        x = self.model(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = 30 * torch.tanh(mean / 30)
        std = self.std_max - F.softplus(self.std_max - std)
        std = self.std_min + F.softplus(std - self.std_min)
        return td.Independent(td.Normal(mean, std), 1)

# Rewards for the next imaginary state
class RewardPrior(nn.Module):
    def __init__(self, latent_dims, hidden_dims, num_actions):
        super().__init__()
        self.action_embedding_dim = 4
        self.action_embedding = nn.Embedding(num_actions, self.action_embedding_dim)

        self.reward = nn.Sequential(
            nn.Linear(latent_dims + self.action_embedding_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1)
        )

        self.apply(torch_utils.weight_init)

    def forward(self, z, a):
        a_embed = self.action_embedding(a)
        z_a = torch.cat([z, a_embed], dim=-1)
        return self.reward(z_a)

class Discriminator(nn.Module):
    def __init__(self, latent_dims, hidden_dims, num_actions):
        super().__init__()
        self.action_embedding_dim = 4
        self.action_embedding = nn.Embedding(num_actions, self.action_embedding_dim)

        input_dim = 2 * latent_dims + self.action_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 2)
        )

        self.apply(torch_utils.weight_init)

    def forward(self, z, a, z_next):
        a_embed = self.action_embedding(a)
        x = torch.cat([z, a_embed, z_next], dim=-1)
        return self.classifier(x)

    def get_reward(self, z, a, z_next):
        logits = self.forward(z, a, z_next)
        reward = logits[..., 1] - logits[..., 0]
        return reward.unsqueeze(-1)
        
class Critic(nn.Module):
    def __init__(self, latent_dims, hidden_dims, num_actions):
        super().__init__()
        self.action_embedding_dim = 4
        self.action_embedding = nn.Embedding(num_actions, self.action_embedding_dim)
        input_dim = latent_dims + self.action_embedding_dim

        self.Q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1)
        )

        self.apply(torch_utils.weight_init)

    def forward(self, z, a):
        a_embed = self.action_embedding(a)
        x_a = torch.cat([z, a_embed], dim=-1)
        q1 = self.Q1(x_a)
        q2 = self.Q2(x_a)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, latent_dims, hidden_dims, num_actions):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, num_actions)  # logits for actions
        )

        self.apply(torch_utils.weight_init)

    def forward(self, z):
        logits = self.policy(z)
        return td.Categorical(logits=logits)
        
class StochasticActor(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, low, high):
        super(StochasticActor, self).__init__()
        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_shape, hidden_dims) 
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2*output_shape)
        self.std_min = np.exp(-5)
        self.std_max = np.exp(2)
        self.apply(torch_utils.weight_init)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = torch.tanh(mean)
        std = self.std_max - F.softplus(self.std_max-std)
        std = self.std_min  + F.softplus(std-self.std_min) 
        dist = torch_utils.TruncatedNormal(mean, std, self.low, self.high)
        return dist