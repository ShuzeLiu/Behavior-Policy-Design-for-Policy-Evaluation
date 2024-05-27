import numpy as np
import torch

from utilities.Torch_utils import *
from utilities.Networks import *

class BasePolicy(object):

    def __init__(self, env):
        self.env = env
        self.prob = None
        return 

    def get_action(self, state):
        pass

    def get_prob(self, state, action):
        pass

    def get_prob_dis(self,state):
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, env):
        BasePolicy.__init__(self, env)
        self.env = env
        self.prob = self.sample_from_simplex_random()
        # self.prob = [0.2, 0.1, 0.1, 0.6]

    def sample_from_simplex(self):
        np.random.seed(0)
        ps = list(sorted(np.random.uniform(size=(self.env.num_actions-1, ))))
        np.random.seed()
        ps = np.array([0] + ps + [1])
        prob = ps[1:] - ps[:-1]
        return prob / np.sum(prob)


    def sample_from_simplex_random(self):
        # np.random.seed(0)
        prob = np.random.uniform(size=(self.env.num_actions, ))
        # np.random.seed()
        return prob / np.sum(prob)



    def get_action(self, state):
        return np.random.choice(self.env.action_space.n,  p=self.prob)

    def get_prob(self, state, action):
        return self.prob[action]
    
    def get_prob_dis(self, state):
        return self.prob

class UniformPolicy(BasePolicy):
    def __init__(self, env):
        BasePolicy.__init__(self, env)
        self.env = env
        self.prob = np.full((self.env.num_actions, ), 1/self.env.num_actions)


    def get_action(self, state):
        return np.random.choice(self.env.action_space.n,  p=self.prob)

    def get_prob(self, state, action):
        return self.prob[action]
    
    def get_prob_dis(self, state):
        return self.prob

class ExtremePolicy(BasePolicy):
    def __init__(self, env):
        BasePolicy.__init__(self, env)
        self.env = env
        self.prob = np.array([0.01,0.01, 0.01, 0.97])


    def get_action(self, state):
        return np.random.choice(self.env.action_space.n,  p=self.prob)

    def get_prob(self, state, action):
        return self.prob[action]
    
    def get_prob_dis(self, state):
        return self.prob
   

class OPEPolicy():
    def __init__(self, env, target_policy, model_w_u, device):
        self.env = env
        self.target_policy = target_policy
        self.model_w_u = model_w_u 
        self.device = device

    def get_action(self, state):
        prob_mu = self.get_prob_dis(state)
        return np.random.choice(self.env.action_space.n,  p=prob_mu)

    def get_prob(self, state, action):
        prob_mu = self.get_prob_dis(state)
        return prob_mu[action]
    
    def get_prob_dis(self, state):
        prob_pi = self.target_policy.get_prob_dis(state)
        self.model_w_u.eval()
        with torch.no_grad():
            hat_u = self.model_w_u(tensor(state, np.float32, self.device))
        hat_u = tensor_to_np(hat_u)
        hat_u = np.absolute(hat_u)
        prob_mu = np.multiply(prob_pi,   np.sqrt(hat_u))
        for i in range(4):
            if prob_mu[i] == 0:
                return prob_pi
        prob_mu = prob_mu/sum(prob_mu)

        return prob_mu





class DifferentiablePolicy(BasePolicy):
    def __init__(self, config, dis):
        BasePolicy.__init__(self, config.env)
        self.env = config.env
        self.actor = GridActionNetwork(config, dis)


    def get_action(self, state):
        dis =self.get_prob_dis(state)
        dis_np = dis.detach().numpy()
        return np.random.choice(self.env.action_space.n, p = dis_np)

    def get_prob(self, state, action):
        with torch.no_grad():
            dis = self.get_prob_dis(state)

        return dis[action].item()
    
    def get_prob_dis(self, state):
        return self.actor(state)


