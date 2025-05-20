import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.gridworld import GridWorld
from alm import AlmAgent
from utils.visualization import plot_metrics, visualize_policy, plot_trajectories, plot_state_visitation_heatmap

import random
import torch
import time
import wandb
import numpy as np

from pathlib import Path

wandb.login()

wandb.init("kmor-perceptualpolicy-initial")

device = 'cuda'
seed = 1

# Keeping base configuration intact
num_train_steps = 500000
explore_steps = 5000

# Longest route without revisiting a cell is 99 steps.
max_episode_steps = 100
env_buffer_size = 100000
batch_size = 512
seq_len = 3

#learning
gamma = 0.99
tau = 0.005

# Update the target ideally every 1000 steps
target_update_interval = 10
lambda_cost = 0.1
lr = {'model' : 0.0001, 'reward' : 0.0001, 'critic' : 0.0001, 'actor' : 0.0001} #originally 0.0001
max_grad_norm =  100.0

#exploration
expl_start = 1.0
expl_end = 0.25
expl_duration = 100000
stddev_clip = 0.25

#hidden_dims and layers. We don't need latent dimensions more than 8 since each state is only (x,y)
latent_dims = 8
hidden_dims = 16    
model_hidden_dims = 32

#bias evaluation
eval_bias = False 
eval_bias_interval = 500

#evaluation
eval_episode_interval = 5000
num_eval_episodes = 5

#saving
save_snapshot = False
save_snapshot_interval = 50000

wandb_log = True

def make_agent(env, device):    

    num_states = np.prod(env.observation_space.shape[0])
    num_actions = np.prod(env.action_space.n)
    action_low = env.observation_space.low
    action_high = env.observation_space.high

    env_buffer_size = 100000
    buffer_size = min(env_buffer_size, num_train_steps)

    agent = AlmAgent(device, action_low, action_high, num_states, num_actions,
                     buffer_size, gamma, tau, target_update_interval,
                     lr, max_grad_norm, batch_size, seq_len, lambda_cost,
                     expl_start, expl_end, expl_duration, stddev_clip, 
                     latent_dims, hidden_dims, model_hidden_dims,
                     log_wandb = True, log_interval=500)

    return agent

class ALM_Helper:
    def __init__(self, env):
        self.work_dir = Path.cwd()
        self.device = torch.device("cpu") #can be changed to cuda as required.
        self.set_seed()
        self.train_env = env
        self.eval_env = env
        self.agent = make_agent(self.train_env, self.device)
        self._train_step = 0
        self._train_episode = 0
        self._best_eval_returns = -np.inf

    def set_seed(self):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _explore(self):
        state, done = self.train_env.reset(), False
        
        for _ in range(1, explore_steps):
            action = self.train_env.action_space.sample()
            next_state, reward, done, info, _, _ = self.train_env.step(action)
            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            if done:
                state, done = self.train_env.reset(), False
            else:
                state = next_state
            
    def train(self):
        self._explore()
        self._eval()

        state, done, episode_start_time = self.train_env.reset(), False, time.time()
        returns = 0
        
        for _ in range(1, num_train_steps-explore_steps+1):

            action = self.agent.get_action(state, self._train_step)

            action = int(np.argmax(action))

            next_state, reward, done, info, new_distance, prev_distance = self.train_env.step(action)
            self._train_step += 1

            self.agent.env_buffer.push((state, action, reward, next_state, False if info.get("TimeLimit.truncated", False) else done))

            self.agent.update(self._train_step)

            if (self._train_step)%eval_episode_interval==0:
                self._eval()

            if save_snapshot and (self._train_step)%save_snapshot_interval==0:
                self.save_snapshot()

            if done:
                self._train_episode += 1
                returns += info["episode"]["r"]
                print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(info["episode"]["r"], 2)))
                print("Cumulative Returns:", returns)
                if wandb_log:
                    episode_metrics = dict()
                    episode_metrics['episodic_length'] = info["episode"]["l"]
                    episode_metrics['episodic_return'] = info["episode"]["r"]
                    episode_metrics['steps_per_second'] = info["episode"]["l"]/(time.time() - episode_start_time)
                    episode_metrics['env_buffer_length'] = len(self.agent.env_buffer)
                    episode_metrics['new_distance'] = new_distance
                    episode_metrics['prev_distance'] = prev_distance
                    wandb.log(episode_metrics, step=self._train_step)
                state, done, episode_start_time = self.train_env.reset(), False, time.time()
            else:
                state = next_state

        self.train_env.close()
    
    def _eval(self):
        returns = 0 
        steps = 0
        for _ in range(num_eval_episodes):
            done = False 
            state = self.eval_env.reset()
            while not done:
                action = self.agent.get_action(state, self._train_step, True)

                action = int(np.argmax(action))

                next_state, _, done ,info, _, _ = self.eval_env.step(action)
                state = next_state
                
            returns += info["episode"]["r"]
            steps += info["episode"]["l"]
            
            print("Episode: {}, total numsteps: {}, return: {}".format(self._train_episode, self._train_step, round(info["episode"]["r"], 2)))

        eval_metrics = dict()
        eval_metrics['eval_episodic_return'] = returns/num_eval_episodes
        eval_metrics['eval_episodic_length'] = steps/num_eval_episodes

        if save_snapshot and returns/num_eval_episodes >= self._best_eval_returns:
            self.save_snapshot(best=True)
            self._best_eval_returns = returns/num_eval_episodes

        if wandb_log:
            wandb.log(eval_metrics, step = self._train_step)
        
    def _eval_bias(self):
        final_mc_list, final_obs_list, final_act_list = self._mc_returns()
        final_mc_norm_list = np.abs(final_mc_list.copy())
        final_mc_norm_list[final_mc_norm_list < 10] = 10

        obs_tensor = torch.FloatTensor(final_obs_list).to(self.device)
        acts_tensor = torch.FloatTensor(final_act_list).to(self.device)
        lower_bound = self.agent.get_lower_bound(obs_tensor, acts_tensor)
        
        bias = final_mc_list - lower_bound
        normalized_bias_per_state = bias / final_mc_norm_list

        if wandb_log:
            metrics = dict()
            metrics['mean_bias'] = np.mean(bias)
            metrics['std_bias'] = np.std(bias)
            metrics['mean_normalised_bias'] = np.mean(normalized_bias_per_state)
            metrics['std_normalised_bias'] = np.std(normalized_bias_per_state)
            wandb.log(metrics, step = self._train_step)

    def _mc_returns(self):
        final_mc_list = np.zeros(0)
        final_obs_list = []
        final_act_list = [] 
        n_mc_eval = 1000
        n_mc_cutoff = 350

        while final_mc_list.shape[0] < n_mc_eval:
            o = self.eval_env.reset()       
            reward_list, obs_list, act_list = [], [], []
            r, d, ep_ret, ep_len = 0, False, 0, 0

            while not d:
                a = self.agent.get_action(o, self._train_step, True)
                obs_list.append(o)
                act_list.append(a)
                o, r, d, _, _, _ = self.eval_env.step(a)
                ep_ret += r
                ep_len += 1
                reward_list.append(r)

            discounted_return_list = np.zeros(ep_len)
            for i_step in range(ep_len - 1, -1, -1):
                if i_step == ep_len -1 :
                    discounted_return_list[i_step] = reward_list[i_step]
                else :
                    discounted_return_list[i_step] = reward_list[i_step] + gamma * discounted_return_list[i_step + 1]

            final_mc_list = np.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
            final_obs_list += obs_list[:n_mc_cutoff]
            final_act_list += act_list[:n_mc_cutoff]

        return final_mc_list, np.array(final_obs_list), np.array(final_act_list)

    def save_snapshot(self, best=False):
        if best:
            snapshot = Path(self.checkpoint_path) / 'best.pt'
        else:
            snapshot = Path(self.checkpoint_path) / Path(str(self._train_step)+'.pt')
        save_dict = self.agent.get_save_dict()
        torch.save(save_dict, snapshot)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = GridWorld(grid_size=10, stochastic=False, noise=0.1, add_obstacles=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    alm_helper = ALM_Helper(env)
    alm_helper.train()

    #plot_metrics(rewards, losses, entropies)
    #visualize_policy(env, agent)
    #plot_trajectories(env, agent, num_trajectories=10, max_steps=100)
    #plot_state_visitation_heatmap(env, agent, max_steps=100)