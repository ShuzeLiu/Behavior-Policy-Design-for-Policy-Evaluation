import os
import gymnasium as gym
import numpy as np
import torch
import pickle
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from gymnasium.wrappers.time_limit import TimeLimit
from utilities.Torch_utils import *

class GridWorld(gym.Env):
    def __init__(self, feature_type = 'tabular', 
        random_action_prob=0.1,
        width=3,
        t_max=5,
        normalize_reward=True,
        offline_data_number = 3000
        ):


        self.width = width 
        self.t_max = t_max
        self.num_actions = 4
        self.action_space = Discrete(self.num_actions)
        self.rewards = {}
        np.random.seed(0)
        max_reward = 0
        for i in range(self.width):
            for j in range(self.width):
                for a in range(self.num_actions):
                    self.rewards[(i, j, a)] = np.random.uniform() * (j* 20 + i * 10 + 1)
                    max_reward = max(self.rewards[(i, j, a)], max_reward)

        if normalize_reward:
            for k in self.rewards.keys():
                self.rewards[k] = self.rewards[k] / max_reward
        np.random.seed()

        self.feature_type = feature_type 
        if self.feature_type == 'tabular':
            self.num_states = self.width * self.width * self.t_max + 1
            self.observation_space = Box(-10, -10, (self.num_states, ))
        elif self.feature_type == 'pseudo_tabular':
            self.num_states = self.width * self.width + 2
            self.observation_space = Box(-10, -10, (self.num_states, ))
        elif self.feature_type == 'linear':
            self.num_states = self.width * self.width
            np.random.seed(0)
            self.phi = np.random.rand(self.num_states, self.num_states // 2)
            np.random.seed()
            self.observation_space = Box(0, 1, (self.num_states // 2 + 1, ))
        else:
            raise NotImplementedError

        self.pos = (0, 0)
        self.t = 0
        self.dxy = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.random_action_prob = random_action_prob
        
        np.random.seed(0)
        self.offline_data_number = offline_data_number
        self.offline_data = self.sample_offline_data()
        np.random.seed()


    def one_hot_vector(self, i):
        temp = np.zeros(self.num_states) 
        temp[i] = 1
        return temp
    
    def state_to_phi(self, s):
        x,y,t = s[0], s[1], s[2]

        if t == self.t_max:
            return  self.one_hot_vector(self.num_states-1)
        if self.feature_type == 'tabular':
            encoded = t * self.width * self.width + x * self.width + y
            return self.one_hot_vector(encoded)
        elif self.feature_type == 'pseudo_tabular':
            t = float(t) / self.t_max
            phi = self.one_hot_vector(self.width * x + y)
            phi[-2] = t
            return phi
        elif self.feature_type == 'linear':
            t = float(t) / self.t_max
            phi = self.phi[self.width * x + y]
            phi = np.concatenate([phi, [t]])
            return phi
    
    def state_list_to_phi_list_device(self, a, device):
        temp = []
        for s in a:
            temp.append(self.state_to_phi(s))
        temp = tensor(temp, np.float32, device)
        return temp
    
    def state_to_phi_device(self, s, device):
        return tensor(self.state_to_phi(s), np.float32, device)


    def reset(self):
        x = self.width // 2
        y = x
        self.pos = (x, y)
        self.t = 0
        return (x,y,self.t), None
    
    def is_legal_pos(self, pos):
        x, y = pos
        return 0 <= x <= self.width - 1 and 0 <= y <= self.width - 1

    def step(self, action):
        x, y = self.pos
        reward = self.rewards[(x, y, action)]

        if np.random.rand() < self.random_action_prob:
            action = np.random.randint(0, len(self.dxy))
        dx, dy = self.dxy[action]     
        next_pos = (x + dx, y + dy)
        if not self.is_legal_pos(next_pos):
            next_pos = self.pos
        
        self.t += 1
        self.pos = next_pos
        done = self.t >= self.t_max
        if done:
            next_state = (0, 0, self.t)
        else:
            next_state = (self.pos[0], self.pos[1], self.t)
        return next_state, reward, done, done, {}

    def simulate_step(self, x, y, t, action):
        reward = self.rewards[(x, y, action)]
        
        if np.random.rand() < self.random_action_prob:
            action = np.random.randint(0, len(self.dxy))
        dx, dy = self.dxy[action]
        next_pos = (x + dx, y + dy)
        if not self.is_legal_pos(next_pos):
            next_pos = (x,y)
        
        return next_pos[0], next_pos[1], t+1, reward



    def sample_offline_data(self):
        data = []
        for _ in range(self.offline_data_number):
            x = np.random.randint(self.width)
            y = np.random.randint(self.width)
            t = np.random.randint(self.t_max)
            a = np.random.randint(self.num_actions)
            nx,ny,nt,r = self.simulate_step(x,y,t,a)

            # vector_s = self.state_to_phi( (x,y) , t )
            # vector_ns = self.state_to_phi( (nx,ny) , nt )
            s = (x,y,t)
            ns = (nx,ny,nt)
            if nt >= self.t_max:
                done = 1
            else:
                done = 0
            data.append( [s,a,r,ns,done] )

        return data
