import numpy as np
import collections
import wandb
import torch
import copy
import random
from agents.BaseAgent import BaseAgent



class TabularVMCAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        #remind what need to use in this agent.
        self.eval_type = config.eval_type
        self.agent_type = config.agent_type
        self.agent_name = config.agent_name
        self.env = config.env
        self.num_train_episode = config.num_train_episode
        self.behavior_policy = config.behavior_policy
        self.target_policy = config.target_policy
        self.wandb_switch = config.wandb_switch
        self.is_truth_agent = config.is_truth_agent
        self.device = config.device
        self.time_feature = config.time_feature

        if not self.is_truth_agent:
            self.truth_value = config.truth_value

        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_step = []
        self.action_counter = collections.Counter()

    # def action_to_discrete(self, env, action, division_number):
    #     action = np.clip(action, env.action_space.low, env.action_space.high)
    #     interval = (env.action_space.high[0] - env.action_space.low[0]) / division_number
    #     action_on_d1 = action[0]
    #     action_on_d1 = min( torch.div((action_on_d1 - env.action_space.low[0]), interval, rounding_mode='trunc'), torch.tensor(division_number - 1))
    #     #get item of action_on_d1
    #     action_on_d1 = int(action_on_d1.item())
    #     # print(action_on_d1)
    #     return action_on_d1


    def run_one_episode(self,i):

        env = self.env
        init_state, _ = env.reset()
        state = init_state
        done = False
        acc_reward = 0
        acc_rho = 1
        t = 0
        
        while not done:
            t += 1 
            with torch.no_grad():
                if self.agent_type == "discretization":
                    # print("discretization")
                    if self.behavior_policy.model_w_u == None:
                        action, action_index = self.behavior_policy.get_action_and_index(state)
                    else:
                        if self.time_feature:
                            action, action_index = self.behavior_policy.get_action_and_index(state, actual_policy = ("improved", t))
                        else:
                            action, action_index = self.behavior_policy.get_action_and_index(state, actual_policy = ("improved"))
                    # print(type(action_index),action_index)
                else:
                    action = self.behavior_policy.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if self.eval_type == "OFF":
                    if self.agent_type == "discretization":
                        if self.behavior_policy.model_w_u == None:
                            acc_rho = acc_rho * self.target_policy.get_prob(state, action, action_index)/self.behavior_policy.get_prob(state, action, action_index)
                        else:
                            if self.time_feature:
                                acc_rho = acc_rho * self.target_policy.get_prob(state, action, action_index)/self.behavior_policy.get_prob(state, action, action_index, actual_policy = ("improved", t))
                            else:
                                acc_rho = acc_rho * self.target_policy.get_prob(state, action, action_index)/self.behavior_policy.get_prob(state, action, action_index, actual_policy = ("improved"))
                        # print("discretization", t, action_index, acc_rho)
                    else:
                        acc_rho = acc_rho * self.target_policy.get_prob(state, action)/self.behavior_policy.get_prob(state, action)    
                else:
                    acc_rho = 1
            acc_reward += reward * acc_rho
            
            state = next_state
            
        env.close()

        self.list_v.append(acc_reward)
        self.list_step.append(t)
        #change list_v to an np array then get the average
        # if self.list_estimate:
        #     self.list_estimate.append((self.list_estimate[-1]*len(self.list_estimate)+acc_reward)/len(self.list_v)  )
        # else:
        #     self.list_estimate.append(acc_reward)

        if self.wandb_switch: 
            if self.is_truth_agent:
                if (i) % 50000 == 0:
                    #use np to average self.list_estimate
                    print(self.agent_type, i)
                    wandb.log({f"acc_reward_avg_{self.eval_type}_{self.config.agent_name}": np.average(self.list_v), \
                            f"acc_step_avg_{self.eval_type}_{self.config.agent_name}": np.average(self.list_step), \
                            f"episodes_{self.eval_type}_{self.config.agent_name}": i})             
        # if (i) % 100 == 0:
        #     #use np to average self.list_estimate
        #     print(self.list_v)
        #     print(self.list_step)
        #     print({"acc_reward_avg": np.average(self.list_v), \
        #             "acc_step_avg": np.average(self.list_step), \
        #             "episodes": i})             

        if not self.is_truth_agent:
            self.list_estimate.append(np.average(self.list_v))
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))


    def run_all_episode(self):
        for i in range(1, self.num_train_episode+1):
            # if i % 100 == 0:
                # print(i)
            self.run_one_episode(i)



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
            
        self.optimizer = torch.optim.Adam(self.behavior_policy.parameters(), lr=config.learning_rate)
        self.batch_size = config.batch_size
        self.pre_train_number = config.pre_train_number
        # print(self.behavior_policy.parameters())

        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_step = []
        self.list_importance_sampling_ratio = []
        self.list_importance_sampling_ratio_avg = []

        self.list_policy_loss = []
        

    def run_one_episode(self, current_episode):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        done = False
        acc_reward = 0
        acc_rho = 1
        t = 0

        trajectory = []
        while not done:
            action, action_log_prob = self.behavior_policy.get_action_and_value(state)[:2]
            # print("action_log_prob0", action_log_prob)
            action_log_prob = action_log_prob[0]
            # print("action", action)
            # print("action_log_prob", action_log_prob)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward, next_state, action_log_prob))
            acc_rho = (acc_rho * self.target_policy.get_prob(state, action)/self.behavior_policy.get_prob(state, action)).item()   
            acc_reward += reward
            state = next_state
            t += 1
        # print(acc_rho)
        env.close()
        acc_reward *= acc_rho

        ### Update policy
        policy_loss = torch.hstack([i[-1] * pow(acc_reward,2) for i in trajectory])
        # print(policy_loss)
        policy_loss = - policy_loss.sum(dim = 0)

        
        self.list_policy_loss.append(policy_loss)
        if current_episode % self.batch_size == 0:
            # print("episode", current_episode)
            # print(policy_loss)
            # print(self.list_policy_loss)
            total_policy_loss = torch.hstack([i for i in self.list_policy_loss]).sum(dim = 0)
            # print(total_policy_loss)
            self.optimizer.zero_grad()
            total_policy_loss.backward()
            self.optimizer.step()

            self.list_policy_loss = []
            # if current_episode == 4:
            #     exit()

        if current_episode > self.pre_train_number:
            self.list_v.append(acc_reward)
            self.list_step.append(t)
            self.list_estimate.append(np.average(self.list_v))
            self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))
            self.list_importance_sampling_ratio.append(acc_rho)
            self.list_importance_sampling_ratio_avg.append(np.average(np.array(self.list_importance_sampling_ratio)))



    def run_all_episode(self):
        for current_episode in range(1, self.num_train_episode + 1 + self.pre_train_number):
            self.run_one_episode(current_episode)

class ROSMCAgent(BaseAgent):
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
        
        self.learning_rate = config.learning_rate

        self.list_v = []  
        self.list_estimate = []
        self.list_error = [] 
        self.list_step = []
        self.list_importance_sampling_ratio = []
        self.list_importance_sampling_ratio_avg = []

        # self.gradient_list = []
        self.total_data_tuple = 0
        self.total_gradient = None

    def run_one_episode(self, current_episode):
        env = self.env
        init_state, _ = env.reset()
        state = init_state
        done = False
        acc_reward = 0
        acc_rho = 1
        t = 0
 
        
        while not done:
            # print("current_episode", current_episode, "t", t)

            self.behavior_policy = copy.deepcopy(self.target_policy)

            if self.learning_rate >= 0:
                state_dict = self.behavior_policy.state_dict()
                if self.total_gradient != None:
                    for k, total_single_parameter_gradient in zip(state_dict.keys(), self.total_gradient):
                        # if k == "actor_logstd":
                        #     continue
                        if total_single_parameter_gradient != None:
                            state_dict[k] = state_dict[k] - total_single_parameter_gradient * self.learning_rate
                        
                        # if k == "actor_logstd":
                        #     print(k, state_dict[k], total_single_parameter_gradient, total_single_parameter_gradient * self.learning_rate)
                        #     print(k, self.target_policy.state_dict()[k])
                self.behavior_policy.load_state_dict(state_dict)


            action, _ = self.behavior_policy.get_action_and_value(state)[:2]


            if self.learning_rate >= 0:
                _, action_log_prob = self.target_policy.get_action_and_value(state, action)[:2]
                #start ROS ----------------------------------------------
                action_log_prob = action_log_prob[0]
                #get the gradient of action_log_prob for self.target_policy
                cur_gradient = torch.autograd.grad(action_log_prob, self.target_policy.parameters(), allow_unused=True)
                new_cur_gradient = []
                for single_parameter_gradient in cur_gradient:
                    if single_parameter_gradient != None:
                        new_cur_gradient.append(torch.clip(single_parameter_gradient, -1e-06, 1e-06))
                    else:
                        new_cur_gradient.append(None)
                cur_gradient = new_cur_gradient
                # for k in cur_gradient:
                #     if k != None:
                #         # print(k.shape)
                #         #print the max in tensor k
                #         print("cur", k.shape, k.max())
                #     else:
                #         print(k)
                
                self.total_data_tuple += 1
                # self.gradient_list.append(gradient)
                # print(gradient)
                # for k in gradient:
                #     print(k.shape)

                # for k in state_dict.keys():
                #     print(k)
                # print(type(state_dict))
                # print(type(gradient))
                # for k,update in zip(state_dict.keys(),gradient):
                #     if update != None:
                #         print(state_dict[k].shape, update.shape)

                # for gradient in gradient_list:
                if self.total_gradient == None:
                    self.total_gradient = cur_gradient
                else:
                    new_total_gradient = []
                    for total_single_parameter_gradient, single_parameter_gradient in zip(self.total_gradient,cur_gradient):
                        if total_single_parameter_gradient != None:
                            new_total_gradient.append(( (self.total_data_tuple -1) / self.total_data_tuple) * total_single_parameter_gradient + (1 / self.total_data_tuple) * single_parameter_gradient)
                        else:
                            new_total_gradient.append(None)
                    self.total_gradient = new_total_gradient

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            acc_rho = 1
            acc_reward += reward
            state = next_state
            t += 1
        # print(acc_rho)
        env.close()
        acc_reward *= acc_rho

        
        self.list_v.append(acc_reward)
        self.list_step.append(t)
        self.list_estimate.append(np.average(self.list_v))
        self.list_error.append(abs(self.truth_value-self.list_estimate[-1]))
        self.list_importance_sampling_ratio.append(acc_rho)
        self.list_importance_sampling_ratio_avg.append(np.average(np.array(self.list_importance_sampling_ratio)))



    def run_all_episode(self):
      
        for current_episode in range(1, self.num_train_episode + 1):
            self.run_one_episode(current_episode)
