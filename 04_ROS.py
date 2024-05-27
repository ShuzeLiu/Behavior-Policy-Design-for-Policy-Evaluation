import csv
import sys
import wandb
import yaml
import gymnasium as gym
import time
import pickle
import torch
import random

import utilities
import environments
import agents

from utilities.Policy import *
from agents import *

#Todo:
#1. check to remove termination state
#2. sample data from random policy
#3. in the data_format add the unchanged action to the observation
#4. check if reward is float32
#5. change the data counter
#7. when the reward is negative add abs on the compare step
#Used to debug

wandb_switch = True
# wandb_switch = True
wandb_sweep_id = "ROS_fixed"

# wandb_sweep_id = "test"


if wandb_switch:
    #CHANGE THIS!
    with open('./06_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    wandb_env_id = wandb.config.env_id
    wandb_policy_index = wandb.config.policy_index
    wandb_learning_rate = wandb.config.learning_rate
    wandb_repeat = wandb.config.a_repeat
    num_train_episode = 2000
else:
    # wandb_env_id = "InvertedPendulum-v4"
    wandb_env_id = "Hopper-v4"
    wandb_policy_index = 1
    # wandb_learning_rate = 10000
    # wandb_learning_rate = -3
    wandb_learning_rate = 0
    wandb_repeat = 1
    num_train_episode = 10

with open("./old_data_mujoco/03_trained_OPE_model_7_env_100_policy_dic/02_working_policy_id_dic", 'rb') as fp:
    working_policy_id_dic = pickle.load(fp)
wandb_policy_id = working_policy_id_dic[wandb_env_id][wandb_policy_index]
print("wandb_policy_id:", wandb_policy_id)

tag =  "|".join(\
    # ["debug_off_number"]+\ 
                [wandb_env_id[:6]] + ["pid_" + str(wandb_policy_id)] + ["re_" + str(wandb_repeat)]  + ["lr" + str(wandb_learning_rate)]+ ["wandb_switch_" + str(wandb_switch)] + ["sweep_" + str(wandb_sweep_id)] )
print("tag:", tag)
if wandb_switch:
    wandb.log({"tag": tag})
    

env = make_env(wandb_env_id)
ppo_agent = PPOMujocoAgent(env)
# ppo_agent.load_state_dict(torch.load(f"constructed_policy/InvertedPendulum-v4/ppo_continuous_action_{wandb_policy_id}.cleanrl_model", map_location=torch.device('cpu')))
ppo_agent.load_state_dict(torch.load(f"constructed_policy/{wandb_env_id}/ppo_continuous_action_{wandb_policy_id}.cleanrl_model", map_location=torch.device('cpu')))
# off_agent = OffMujocoAgent(env, ppo_agent)

target_policy = ppo_agent
with open(f'data_mujoco/02_ground_truth_value/10_policy_avg', 'rb') as fp:
    truth_dic = pickle.load(fp)
truth_value = truth_dic[(wandb_env_id, wandb_policy_id, "reward")]
truth_step = truth_dic[(wandb_env_id, wandb_policy_id, "step")]
print(f"({wandb_env_id}, {wandb_policy_id}, reward)", truth_value)
print(f"({wandb_env_id}, {wandb_policy_id}, step)", truth_step)

# with open(f'old_data_mujoco/02_ground_truth_value/PPO_7_env_50000_avg', 'rb') as fp:
#     truth_dic = pickle.load(fp)
# truth_value = truth_dic[(wandb_env_id, wandb_policy_id, "reward")]
# truth_step = truth_dic[(wandb_env_id, wandb_policy_id, "step")]
# print(f"({wandb_env_id}, {wandb_policy_id}, reward)", truth_value)
# print(f"({wandb_env_id}, {wandb_policy_id}, step)", truth_step)

config_OPE = utilities.Config()
config_OPE.eval_type = "ON"
config_OPE.agent_type = "ROS_continuous"
config_OPE.agent_name = "ROS"
config_OPE.device = 'cpu'
# config_OPE.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config_OPE.device = torch.device('cuda:'+str(best_gpu()) if torch.cuda.is_available() else 'cpu')
config_OPE.repeat = wandb_repeat
config_OPE.policy_id = wandb_policy_id
config_OPE.env_id = wandb_env_id
config_OPE.env = env
config_OPE.num_train_episode = num_train_episode
config_OPE.target_policy = target_policy
config_OPE.behavior_policy = target_policy
config_OPE.is_truth_agent = False
config_OPE.truth_value = truth_value

config_OPE.wandb_switch = wandb_switch
config_OPE.learning_rate = wandb_learning_rate
config_OPE.tag = tag
config_OPE.wandb_sweep_id = wandb_sweep_id


OPE_MC_agent = agents.ROSMCAgent(config_OPE)
OPE_MC_agent.run_all_episode()



config_on_MC = utilities.Config()
config_on_MC.eval_type = "ON"
config_on_MC.agent_type = "continuous"
config_on_MC.agent_name = "PPO_continuous"
config_on_MC.wandb_switch = wandb_switch
config_on_MC.device = 'cpu'
config_on_MC.env = env
config_on_MC.num_train_episode = num_train_episode
config_on_MC.target_policy = ppo_agent
config_on_MC.behavior_policy = ppo_agent
config_on_MC.is_truth_agent = False
config_on_MC.truth_value = truth_value



on_policy_MC_agent = agents.TabularVMCAgent(config_on_MC)
on_policy_MC_agent.run_all_episode()

print("compare start")

acc_OPE_error = 0
acc_on_error = 0
var_OPE = 0
var_on = 0
for i in range(num_train_episode):
    OPE_error = OPE_MC_agent.list_error[i]
    acc_OPE_error += OPE_error
    var_OPE += pow(OPE_error, 2)

    on_error = on_policy_MC_agent.list_error[i]
    acc_on_error +=  on_error   
    var_on += pow(on_error, 2)
    
    if wandb_switch and (i+1) % 500 == 0:
        wandb.log({"normalized_OPE_error": abs(OPE_error/truth_value),   \
            "normalized_on_error": abs(on_error/truth_value), \
            "normalized_error_diff":  abs(on_error/truth_value) -  abs(OPE_error/truth_value), \
            "normalized_error_diff_divide_by_on_error":  (abs(on_error/truth_value) -  abs(OPE_error/truth_value))/ abs(on_error/truth_value)   , \
            "acc_normalized_error_diff":  abs(acc_on_error/truth_value) -  abs(acc_OPE_error/truth_value), \
            "var_ratio": var_OPE/var_on, \
            "normalized_OPE_estimate": OPE_MC_agent.list_estimate[i]/truth_value,\
            "normalized_on_estimate": on_policy_MC_agent.list_estimate[i]/truth_value,\
            "step_within_OPE_episode": OPE_MC_agent.list_step[i],\
            "step_within_on_episode": on_policy_MC_agent.list_step[i],\
            "truth_value": truth_value,\
            "truth_step": truth_step,\
            "step_compare": i+1
            } )


name = config_OPE.tag
# print(name)
p = f"data_mujoco/06_ROS_fixed/{wandb_learning_rate}/{config_OPE.env_id}/"
#check if p exists, otherwise create p
if not os.path.exists(p):
    os.makedirs(p)

dic = {"OPE_list_v": OPE_MC_agent.list_v, "OPE_list_estimate": OPE_MC_agent.list_estimate, "OPE_list_error": OPE_MC_agent.list_error, "OPE_list_step": OPE_MC_agent.list_step,
        # "OPE_list_importance_sampling_ratio": OPE_MC_agent.list_importance_sampling_ratio, "OPE_list_importance_sampling_ratio_avg": OPE_MC_agent.list_importance_sampling_ratio_avg, \
        "on_list_v": on_policy_MC_agent.list_v, "on_list_estimate": on_policy_MC_agent.list_estimate, "on_list_error": on_policy_MC_agent.list_error, "on_list_step": on_policy_MC_agent.list_step, \
        "truth_value": truth_value, "truth_step": truth_step}



with open(p + name, 'wb') as fp:
    pickle.dump(dic, fp)


print("compare end")

if wandb_switch:
    wandb.finish()



