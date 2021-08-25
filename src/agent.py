from typing import Dict, Union

import numpy as np
import torch
from torch.nn import functional as f

from model import Actor, Critic
from per import Annealing, PrioritizedReplayBuffer


class DDPGAgent:
    def __init__(self, action_size: int, state_size: int, optimizers_cls: Dict, beta_init: float,
                 optimizers_kwargs: Dict[str, Dict], batch_size: int, num_steps: int, alpha: float,
                 buffer_size: Union[int, float], device: torch.device, gamma: float,
                 update_every: int, soft_update_step: float, seed: int = 55321):
        
        self.beta = Annealing(beta_init, 1, num_steps)
        self.soft_update_step = soft_update_step
        self.device = device
        self.gamma = gamma
        torch.manual_seed(seed)
        
        self.actor = Actor(state_size=state_size, action_size=action_size, seed=seed,
                           device=device, fc1_units=128, fc2_units=128)
        
        self.actor_target = Actor(state_size=state_size, action_size=action_size, seed=seed,
                                  fc1_units=128, fc2_units=128, device=device)
        self.critic = Critic(state_size=state_size, action_size=action_size, seed=seed,
                             fcs1_units=128, fc2_units=128, device=device)
        self.critic_target = Critic(state_size=state_size, action_size=action_size, seed=seed,
                                    fcs1_units=128, fc2_units=128, device=device)
        
        self.actor_optim = optimizers_cls['actor'](self.actor.parameters(),
                                                   **optimizers_kwargs['actor'])
        self.critic_optim = optimizers_cls['critic'](self.critic.parameters(),
                                                     **optimizers_kwargs['critic'])
        
        self.curr_step = 0
        self.learning_start_threshold = 0000
        self.update_every = update_every
        self.noise = OUNoise(action_size, seed)
        self.noise_decay = Annealing(5e-1, 1e-2, 2e5)
        
        self.memory = PrioritizedReplayBuffer(batch_size, buffer_size, alpha=alpha, seed=seed,
                                              device=device)
    
    def save(self, folder, file_prefix):
        torch.save(self.actor.state_dict(), f"{folder}/{file_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{folder}/{file_prefix}_critic.pth")
    
    def load(self, folder, file_prefix):
        self.actor.load_state_dict(torch.load(f'{folder}/{file_prefix}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{folder}/{file_prefix}_critic.pth'))
    
    def act(self, state: np.ndarray, train: bool = True):
        state = torch.FloatTensor(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0).to(self.device)).cpu()
        self.actor.train()
        if train:
            action += self.noise.sample() * self.noise_decay.step()
        return torch.clamp(action, -1, 1).numpy()
    
    def learn(self):
        states, actions, rewards, next_states, dones, nodes_value, sample_ids = self.memory.sample()
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        
        targets_q = rewards + (self.gamma * next_values * (1 - dones))
        
        expected_q = self.critic(states, actions)
        critic_loss = (expected_q - targets_q).squeeze()
        
        self.memory.update(critic_loss, sample_ids)
        
        weights = self.memory.calc_is_weight(nodes_value, self.beta.step())
        critic_loss = critic_loss.mul(weights)
        critic_loss = f.mse_loss(critic_loss, torch.zeros_like(critic_loss))
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        actions_predicted = self.actor(states)
        actor_loss = -self.critic(states, actions_predicted).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action.squeeze(0), reward, next_state, done)
        
        self.curr_step += 1
        if self.curr_step > self.learning_start_threshold and self.curr_step % self.update_every \
                == 0:
            self.learn()
    
    def reset_noise(self):
        self.noise.reset()
    
    def soft_update(self, local, target):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(
                    self.soft_update_step * l_param.data + (
                            1 - self.soft_update_step) * t_param.data)


class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu: np.ndarray = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
        self.state = self.mu.copy()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(
                size=len(self.state))
        return self.state
