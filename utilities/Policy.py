import numpy as np
import torch
import random

from utilities.Torch_utils import *
from utilities.Networks import *
from torch.distributions.normal import Normal

class BasePolicy(object):

    def __init__(self, env):
        self.env = env
        self.prob = None
        return 

    def get_action(self, state):
        pass

    def get_prob(self, state, action):
        pass

    def get_prob_dis(self, state):
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
        prob_pi = self.target_policy.get_prob_dis(state[:self.env.observation_space.shape[0]])
        self.model_w_u.eval()
        with torch.no_grad():
            hat_u = self.model_w_u(tensor(state, np.float32, self.device))
        hat_u = tensor_to_np(hat_u)
        hat_u = np.absolute(hat_u)
        prob_mu = np.multiply(prob_pi,   np.sqrt(hat_u))
        for i in range(2):
            if prob_mu[i] == 0:
                return prob_pi
        prob_mu = prob_mu/sum(prob_mu)

        return prob_mu




class CartPolePolicy():
    def __init__(self, env, policy_id, device, epsilon):
        BasePolicy.__init__(self, env)
        self.env = env
        self.q_network = QNetwork(env).to(device)
        self.q_network.load_state_dict(torch.load(f'data/cart_pole/01_cart_pole_policy_0/cart_pole_0_{policy_id}', map_location=device))
        self.q_network.eval()

        self.device = device
        self.epsilon = epsilon

    def get_action(self, state):
        prob_mu = self.get_prob_dis(state)
        action = np.random.choice(self.env.action_space.n,  p=prob_mu)
        return action 
    
    def get_prob(self, state, action):
        return self.get_prob_dis(state)[action]
    
    def get_prob_dis(self, state):
        q_values = self.q_network(torch.Tensor(state).to(self.device))
        action = torch.argmax(q_values).cpu().numpy()
        prob = np.array([self.epsilon/2, self.epsilon/2])
        prob[action] = 1-self.epsilon/2
        return prob

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOMujocoAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = np.asarray(x, dtype=np.float32)
        x = tensor([x], np.float32, "cpu")
        action_mean = self.actor_mean(x)
        # print("action_mean", action_mean)      
        # print("action_logstd", self.actor_logstd)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # print("action_mean", action_mean)
        # print("action_logstd", action_logstd)
        # print("action_std", action_std)
        probs = Normal(action_mean, action_std)
        # print("probs", probs)
        if action is None:
            action = probs.sample()
        else:
            action = tensor([action], np.float32, "cpu")
        # print("action", action)
        # print("probs.log_prob(action)", probs.log_prob(action))
        # print("probs.log_prob(action).sum(1)", probs.log_prob(action).sum(1))
        # print("probs.log_prob(action).sum(1)[0]", probs.log_prob(action).sum(1)[0])
        prob = torch.exp(probs.log_prob(action).sum(1))
        return action[0].numpy(), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), prob[0]

    def get_action(self, state):
        return self.get_action_and_value(state)[0]
    
    def get_prob(self, state, action):
        return self.get_action_and_value(state, action)[-1]

class OffMujocoAgent():
    def __init__(self, env, ppo_agent):
        self.env = env
        self.ppo_agent = ppo_agent
        self.model_w_u = None
        self.division_number = 10

    def calculate_action_prob(self, state, action, changed_action_prob, actual_policy):
        #tesnor, tesnor, tesnor
        if len(actual_policy) == 2:
            time_step = tensor([actual_policy[1]], np.float32, "cpu")
            state_action = torch.cat((time_step, state, action[1:]))
        else:
            state_action = torch.cat((state, action[1:]))
        #This is very import!
        #If the convention is not consistent, get the first element [0] causes disaster
        state_action = state_action.unsqueeze(0)
        self.model_w_u.eval()
        with torch.no_grad():
            hat_u = self.model_w_u(state_action)            
        hat_u = hat_u.abs().sqrt()[0]
        
        old_changed_action_prob = changed_action_prob
        changed_action_prob = torch.mul(hat_u, changed_action_prob)
        changed_action_prob = changed_action_prob / changed_action_prob.sum()
        for i in range(self.division_number):
            if changed_action_prob[i] == 0:
                changed_action_prob = old_changed_action_prob
                break
        #return a tensor
        return changed_action_prob


    #input must be an np array
    def get_action_and_value(self, x, action = None, action_index = None, actual_policy = "original"):
        x = tensor([x], np.float32, "cpu")
        action_mean = self.ppo_agent.actor_mean(x)
        action_logstd = self.ppo_agent.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_mean, action_std = action_mean[0], action_std[0] 


        changed_probs = Normal(action_mean[0], action_std[0])      
        interval = (self.env.action_space.high[0] - self.env.action_space.low[0]) / self.division_number
        division_end = [self.env.action_space.low[0] + interval * i for i in range(1, self.division_number + 1)]
        division_cdf = changed_probs.cdf(torch.tensor(division_end))
        changed_action_prob = [division_cdf[0]]
        for i in range(self.division_number - 2):
            changed_action_prob.append(division_cdf[i+1] - division_cdf[i])
        changed_action_prob.append(1 - division_cdf[-2])
        changed_action_prob = torch.tensor(changed_action_prob)
        changed_action_prob = changed_action_prob / changed_action_prob.sum()

        
        if action_index == None:
            combined_action = Normal(action_mean, action_std).sample()
            if actual_policy != "original":
                changed_action_prob = self.calculate_action_prob(x[0], combined_action, changed_action_prob, actual_policy)
            changed_action_index = torch.multinomial(changed_action_prob, 1)[0].item()
            combined_action[0] =  self.env.action_space.low[0] + interval * changed_action_index + interval / 2
            combined_action = combined_action.numpy()
        else:
            combined_action = action
            if actual_policy != "original":
                changed_action_prob = self.calculate_action_prob(x[0], tensor(combined_action, np.float32, "cpu"), changed_action_prob, actual_policy)
            changed_action_index = action_index

        
        changed_action_sample_prob = changed_action_prob[changed_action_index]
        changed_action_sample_prob = changed_action_sample_prob.item() 
        # print(combined_action, changed_action_index, changed_action_sample_prob)
        # exit()
        #output: numpy arrary, int, float
        return combined_action, changed_action_index, changed_action_sample_prob



    def get_action_and_index(self, state, actual_policy = "original"):
        combined_action, changed_action_index, _ = self.get_action_and_value(state, actual_policy = actual_policy)
        return combined_action, changed_action_index

    def get_prob(self, state, action, action_index, actual_policy = "original"):
        return self.get_action_and_value(state, action, action_index, actual_policy = actual_policy)[-1]
    
class OffMujocoAgentNoise():
    def __init__(self, env, ppo_agent, max_epsilon):
        self.env = env
        self.ppo_agent = ppo_agent
        self.model_w_u = None
        self.division_number = 10
        self.max_epsilon = max_epsilon
        self.cur_epsilon = random.uniform(0, self.max_epsilon)
    
    # def calculate_action_prob(self, state, action, changed_action_prob):
    #     #tesnor, tesnor, tesnor
    #     state_action = torch.cat((state, action[1:]))
    #     #This is very import!
    #     #If the convention is not consistent, get the first element [0] causes disaster
    #     state_action = state_action.unsqueeze(0)
    #     self.model_w_u.eval()
    #     with torch.no_grad():
    #         hat_u = self.model_w_u(state_action)            
    #     hat_u = hat_u.abs().sqrt()[0]
        
    #     old_changed_action_prob = changed_action_prob
    #     changed_action_prob = torch.mul(hat_u, changed_action_prob)
    #     changed_action_prob = changed_action_prob / changed_action_prob.sum()
    #     for i in range(self.division_number):
    #         if changed_action_prob[i] == 0:
    #             changed_action_prob = old_changed_action_prob
    #             break
    #     #return a tensor
    #     return changed_action_prob

    def reset_epsilon(self):
        #sample a number from [0,10]
        n = random.randint(1, 30)
        self.cur_epsilon = self.max_epsilon * (n / 30)

        # self.cur_epsilon = random.uniform(0, self.max_epsilon)

    #input must be an np array
    def get_action_and_value(self, x, action = None, action_index = None, actual_policy = "original"):
        x = tensor([x], np.float32, "cpu")
        action_mean = self.ppo_agent.actor_mean(x)
        action_logstd = self.ppo_agent.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        action_mean, action_std = action_mean[0], action_std[0] 


        changed_probs = Normal(action_mean[0], action_std[0])      
        interval = (self.env.action_space.high[0] - self.env.action_space.low[0]) / self.division_number
        division_end = [self.env.action_space.low[0] + interval * i for i in range(1, self.division_number + 1)]
        division_cdf = changed_probs.cdf(torch.tensor(division_end))
        changed_action_prob = [division_cdf[0]]
        for i in range(self.division_number - 2):
            changed_action_prob.append(division_cdf[i+1] - division_cdf[i])
        changed_action_prob.append(1 - division_cdf[-2])
        changed_action_prob = torch.tensor(changed_action_prob)
        changed_action_prob = changed_action_prob / changed_action_prob.sum()

        
        if action_index == None:
            combined_action = Normal(action_mean, action_std).sample()
            # if actual_policy == "improved":
            #     changed_action_prob = self.calculate_action_prob(x[0], combined_action, changed_action_prob)

            changed_action_index = torch.multinomial(changed_action_prob, 1)[0].item()
            # print(self.cur_epsilon)
            if random.uniform(0, 1) < self.cur_epsilon:
                # print("changed")
                changed_action_index = random.randint(0, self.division_number - 1)

            combined_action[0] =  self.env.action_space.low[0] + interval * changed_action_index + interval / 2
            combined_action = combined_action.numpy()
        else:
            combined_action = action
            if actual_policy == "improved":
                changed_action_prob = self.calculate_action_prob(x[0], tensor(combined_action, np.float32, "cpu"), changed_action_prob)
            changed_action_index = action_index
        # print(self.cur_epsilon )
        
        changed_action_sample_prob = changed_action_prob[changed_action_index]
        changed_action_sample_prob = changed_action_sample_prob.item() 
        # print(combined_action, changed_action_index, changed_action_sample_prob)
        # exit()
        #output: numpy arrary, int, float
        return combined_action, changed_action_index, changed_action_sample_prob



    def get_action_and_index(self, state, actual_policy = "original"):
        combined_action, changed_action_index, _ = self.get_action_and_value(state, actual_policy = actual_policy)
        return combined_action, changed_action_index

    def get_prob(self, state, action, action_index, actual_policy = "original"):
        return self.get_action_and_value(state, action, action_index, actual_policy = actual_policy)[-1]