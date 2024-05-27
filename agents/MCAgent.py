import numpy as np
import collections
import wandb
import torch
import matplotlib.pyplot as plt
import copy

from torch.distributions.categorical import Categorical
from agents.BaseAgent import BaseAgent


class TabularVMCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = config.behavior_policy
        self.target_policy = config.target_policy
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        if not self.is_truth_agent:
            self.truth_value = config.truth_value


        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_importance_sampling_ratio = []
        self.list_importance_sampling_ratio_avg = []


    def run_one_episode(self,i):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        state = env.state_to_phi_device(state, self.device)
        done = False
        acc_reward = 0
        acc_rho = 1
        list_rho = []
        while not done:
            action = self.behavior_policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_to_phi_device(next_state, self.device)
            done = terminated or truncated
            acc_rho = acc_rho * self.target_policy.get_prob(state, action)/self.behavior_policy.get_prob(state, action) 
            list_rho.append(acc_rho) 
            acc_reward += reward * acc_rho
            state = next_state
        env.close()

        self.list_v.append(acc_reward)
        if self.list_estimate:
            self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        else:
            self.list_estimate.append(acc_reward)

        if self.wandb_switch: 
            if self.is_truth_agent:
                if (i) % 10000 == 0:
                    wandb.log({"estimate": self.list_estimate[-1], \
                            "episodes": i})             



        if not self.is_truth_agent:
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))
        
        self.list_importance_sampling_ratio.append(list_rho)



    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            self.run_one_episode(i)



class Tabular_UCB_V_MCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = config.behavior_policy
        self.target_policy = config.target_policy
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        self.c = config.c
        if not self.is_truth_agent:
            self.truth_value = config.truth_value


        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_importance_sampling_ratio = []
        
        self.UCB_reward = [[],[]]
        self.UCB_score = [float("inf"), float("inf")]

    def UCB_choice(self, t):
        #update_UCB_score
        for i, reward in enumerate(self.UCB_reward):
            if len(reward):
                self.UCB_score[i] = np.average(reward) + pow(2, self.config.c) * np.sqrt( np.log(t) / len(reward))

        policy_index = np.argmax(self.UCB_score)
        if policy_index == 0:
            self.actual_behavior_policy = self.config.behavior_policy
        else:
            self.actual_behavior_policy = self.config.target_policy


    def run_one_episode(self,i):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        state = env.state_to_phi_device(state, self.device)
        done = False
        acc_reward = 0
        acc_rho = 1
        list_rho = []
        while not done:
            action = self.actual_behavior_policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_to_phi_device(next_state, self.device)
            done = terminated or truncated
            acc_rho = acc_rho * self.target_policy.get_prob(state, action)/self.actual_behavior_policy.get_prob(state, action)   
            list_rho.append(acc_rho)
            acc_reward += reward * acc_rho
            state = next_state
        env.close()

        self.list_v.append(acc_reward)
        if self.list_estimate:
            self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        else:
            self.list_estimate.append(acc_reward)

        if self.wandb_switch: 
            if self.is_truth_agent:
                if (i) % 10000 == 0:
                    wandb.log({"estimate": self.list_estimate[-1], \
                            "episodes": i})             

        if not self.is_truth_agent:
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))
        
        self.list_importance_sampling_ratio.append(list_rho)

        self.UCB_reward[np.argmax(self.UCB_score)].append(-pow(acc_reward,2))


    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            self.UCB_choice(i)
            self.run_one_episode(i)



class TabularReinforceMCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = config.behavior_policy
        self.target_policy = config.target_policy
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        if not self.is_truth_agent:
            self.truth_value = config.truth_value

        self.optimizer = torch.optim.Adam(self.behavior_policy.actor.parameters(), lr=config.learning_rate)
        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 


    def run_one_episode(self,i):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        state = env.state_to_phi_device(state, self.device)
        done = False
        acc_reward = 0
        acc_rho = 1
        trajectory = []
        while not done:
            action, dis = self.behavior_policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_to_phi_device(next_state, self.device)
            done = terminated or truncated
            trajectory.append((state, action, reward, next_state, dis))
            acc_rho = acc_rho * self.target_policy.get_prob(state, action)/self.behavior_policy.get_prob(state, action)   
            acc_reward += reward * acc_rho
            state = next_state
        env.close()


        self.list_v.append(acc_reward)

        # if self.list_estimate:
        #     self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        # else:
        #     self.list_estimate.append(acc_reward)
        
        # print(torch.log(trajectory[-1][-1]) * acc_reward )
        policy_action = torch.tensor([i[1] for i in trajectory]).view(-1,1)
        policy_loss = torch.vstack([torch.log(i[-1]) * acc_reward for i in trajectory])
        # print(policy_loss.shape)
        # print(policy_loss)
        # print(policy_action)
        policy_loss = policy_loss.gather(1, policy_action)
        # print(policy_loss)
        policy_loss = - policy_loss.sum(dim = 0)
        # print(policy_loss)
        # print(policy_loss.shape)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


        # if not self.is_truth_agent:
        #     self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))



    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            self.run_one_episode(i)

        temp_y = []
        self.list_v = np.array(self.list_v)
        for i in range(len(self.list_v)//10):
            # temp_y.append(np.average(self.list_v[i:i+100]))
            temp_y.append(np.average(self.list_v[:i+1]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(0, int(len(self.list_v)//10) ), temp_y)
        plt.xlabel("episode")
        plt.ylabel("return")
        plt.savefig('reinforce.pdf', bbox_inches='tight')




class TabularBPSMCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = copy.deepcopy(config.target_policy)
        self.target_policy = config.target_policy 
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        if not self.is_truth_agent:
            self.truth_value = config.truth_value
            
        self.optimizer = torch.optim.Adam(self.behavior_policy.actor.parameters(), lr=config.learning_rate)
        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_importance_sampling_ratio = []
        self.list_importance_sampling_ratio_avg = []

    def run_one_episode(self, i):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        state = env.state_to_phi_device(state, self.device)
        done = False
        acc_reward = 0
        acc_rho = 1
        trajectory = []
        while not done:
            action = self.behavior_policy.get_action(state)
            dis = self.behavior_policy.get_prob_dis(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_to_phi_device(next_state, self.device)
            done = terminated or truncated
            trajectory.append((state, action, reward, next_state, dis))
            acc_rho = acc_rho * self.target_policy.get_prob(state, action)/self.behavior_policy.get_prob(state, action)   
            acc_reward += reward
            state = next_state
        env.close()
        acc_reward *= acc_rho
        ### Update policy
        # print(torch.log(trajectory[-1][-1]) * acc_reward )
        policy_action = torch.tensor([i[1] for i in trajectory]).view(-1,1)
        policy_loss = torch.vstack([torch.log(i[-1]) * pow(acc_reward,2) for i in trajectory])
        # print(policy_loss.shape)
        # print(policy_loss)
        # print(policy_action)
        policy_loss = policy_loss.gather(1, policy_action)
        # print(policy_loss)
        policy_loss = - policy_loss.sum(dim = 0)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


        self.list_v.append(acc_reward)
        if self.list_estimate:
            self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        else:
            self.list_estimate.append(acc_reward)

        if not self.is_truth_agent:
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))

        self.list_importance_sampling_ratio.append(acc_rho)
        self.list_importance_sampling_ratio_avg.append(np.average(np.array(self.list_importance_sampling_ratio)))

    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            self.run_one_episode(i)



class TabularROAMCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = copy.deepcopy(config.target_policy)
        self.target_policy = config.target_policy 
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        if not self.is_truth_agent:
            self.truth_value = config.truth_value

        self.learning_rate  = config.learning_rate  
        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_importance_sampling_ratio = []
        self.list_importance_sampling_ratio_avg = []

        self.loss_tensor_list = None

    def run_one_episode(self, i):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        state = env.state_to_phi_device(state, self.device)
        done = False
        acc_reward = 0
        acc_rho = 1
        trajectory = []
        gradient_list = []
        while not done:
            self.behavior_policy = copy.deepcopy(self.target_policy)
            if gradient_list:
                # print(gradient)
                # new_parameters = self.behavior_policy.actor.parameters()
                # print(new_parameters)
                state_dict = self.behavior_policy.actor.state_dict()
                # for k,update  in zip(state_dict.keys(),gradient):
                #     print(state_dict[k],update)
                # print("-------------old------------")

                for gradient in gradient_list:
                    for k,update  in zip(state_dict.keys(),gradient):
                        state_dict[k] = state_dict[k] - update * (self.learning_rate/len(gradient_list))

                self.behavior_policy.actor.load_state_dict(state_dict)

            action = self.behavior_policy.get_action(state)
            dis = self.target_policy.get_prob_dis(state)

            dis = Categorical(logits=dis)
            log_dis = dis.log_prob(torch.tensor(action))
            gradient = torch.autograd.grad(log_dis, self.target_policy.actor.parameters(), allow_unused=True)
            gradient_list.append(gradient)
            # policy_loss = dis.gather(0, action)
            # print(dis, action, policy_loss)
            # gradients = torch.autograd.grad(policy_loss, self.target_policy.actor.parameters())
            # print(gradients)



            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = env.state_to_phi_device(next_state, self.device)
            done = terminated or truncated
            trajectory.append((state, action, reward, next_state, dis))
            acc_rho = 1
            acc_reward += reward
            state = next_state


            

        env.close()
        # acc_reward *= acc_rho



        self.list_v.append(acc_reward)
        if self.list_estimate:
            self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        else:
            self.list_estimate.append(acc_reward)

        if not self.is_truth_agent:
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))

        self.list_importance_sampling_ratio.append(acc_rho)
        self.list_importance_sampling_ratio_avg.append(np.average(np.array(self.list_importance_sampling_ratio)))

    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            self.run_one_episode(i)

