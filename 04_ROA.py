import csv
import sys
import wandb
import yaml
import gymnasium as gym
import time
import pickle
import torch

import utilities
import environments
import agents

from utilities.Policy import *
from utilities.Torch_utils import *
from agents.MCAgent import *
from agents.OPEAgent import *
from environments.GridWorld import GridWorld

wandb_switch = True

if wandb_switch:
    with open('./03_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    wandb_env_size = wandb.config.a_env_size
    wandb_policy_id = wandb.config.policy_id
    wandb_offline_data_number =  wandb.config.offline_data_number
    wandb_learning_rate =  pow(10, wandb.config.learning_rate)
    wandb_feature_type = wandb.config.feature_type
    wandb_repeat = wandb.config.repeat
    wandb_policy_id =  wandb.config.policy_id
    num_train_episode = 50
    wandb_layer_structure = wandb.config.layer_structure
    wandb_global_repeat = wandb.config.aaa_global_repeat
else:
    wandb_env_size = 5
    wandb_policy_id = 1
    wandb_offline_data_number = 1
    wandb_learning_rate = 0.001
    wandb_feature_type = "pseudo_tabular"
    wandb_repeat = 1
    num_train_episode = 1
    wandb_layer_structure = "4"
    wandb_global_repeat = 1


tag =  "|".join(["ROA", str(wandb_env_size), str(wandb_learning_rate), wandb_layer_structure, str(wandb_global_repeat)])
print("tag:", tag)
if wandb_switch:
    wandb.log({"tag": tag})

env = environments.GridWorld(width = wandb_env_size, \
    t_max = wandb_env_size, \
    offline_data_number = wandb_offline_data_number,\
    feature_type = wandb_feature_type
    )


if wandb_env_size < 60:
    with open('data/02_ground_truth', 'rb') as fp:
        dic_truth = pickle.load(fp)
    truth_value = dic_truth[(wandb_env_size, wandb_policy_id)]
else:
    with open('data/large_linear/02_ground_truth', 'rb') as fp:
        dic_truth = pickle.load(fp)
    # print(dic_truth)
    truth_value = dic_truth[(wandb_env_size, wandb_policy_id)]



#Fix parameters



config_OPE = utilities.Config()
config_OPE.feature_type = wandb_feature_type
config_OPE.wandb_layer_structure = wandb_layer_structure
config_OPE.device = 'cpu'
# config_OPE.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config_OPE.device = torch.device('cuda:'+str(best_gpu()) if torch.cuda.is_available() else 'cpu')
# print("-----------------", config_OPE.device)
config_OPE.repeat = wandb_repeat
config_OPE.policy_id = wandb_policy_id
config_OPE.env = env
config_OPE.num_train_episode = num_train_episode
config_OPE.is_truth_agent = False
config_OPE.truth_value = truth_value
config_OPE.learning_rate = wandb_learning_rate
config_OPE.wandb_switch = wandb_switch

torch.set_printoptions(precision=8)


with open('data/01_policy_data', 'rb') as fp:
    list_poilcy = pickle.load(fp)
dis = list_poilcy[wandb_policy_id].get_prob_dis("1")
print("Original Policy: ", dis)

config_OPE.target_policy = DifferentiablePolicy(config_OPE, dis)
config_OPE.behavior_policy = config_OPE.target_policy

# print(target_policy.actor.nn_stack[0].weight)
# print(target_policy.actor.nn_stack[0].bias)

# for i in config_OPE.target_policy.actor.nn_stack:
#     print(i)
# with torch.no_grad():
#     print(config_OPE.target_policy.actor(torch.rand(config_OPE.env.num_states)))
#     print(config_OPE.target_policy.get_action(torch.rand(config_OPE.env.num_states)))


ROA_agent = TabularROAMCAgent(config_OPE)
ROA_agent.run_all_episode()



config_on_MC = utilities.Config()
config_on_MC.wandb_switch = wandb_switch
config_on_MC.device = 'cpu'
config_on_MC.env = env
config_on_MC.num_train_episode = num_train_episode
config_on_MC.target_policy = list_poilcy[wandb_policy_id]
config_on_MC.behavior_policy = list_poilcy[wandb_policy_id]
config_on_MC.is_truth_agent = False
config_on_MC.truth_value = truth_value

on_policy_MC_agent = agents.TabularVMCAgent(config_on_MC)
on_policy_MC_agent.run_all_episode()


print("len(ROA_agent.list_error):", len(ROA_agent.list_error))


acc_OPE_error = 0
acc_on_error = 0
var_OPE = 0
var_on = 0
for i in range(num_train_episode):

    OPE_error = ROA_agent.list_error[i]
    acc_OPE_error += OPE_error
    var_OPE += pow(OPE_error, 2)

    on_error = on_policy_MC_agent.list_error[i]
    acc_on_error +=  on_error   
    var_on += pow(on_error, 2)

    if wandb_switch:
    # if i == num_train_episode - 1 and wandb_switch:
        wandb.log({"normalized_OPE_error": OPE_error/truth_value,   \
            "normalized_on_error": on_error/truth_value, \
            "acc_normalized_error_diff":  acc_on_error/truth_value -  acc_OPE_error/truth_value, \
            "normalized_error_diff":  on_error/truth_value -  OPE_error/truth_value, \
            "var_ratio": var_OPE/var_on, \
            "normalized_OPE_estimate": ROA_agent.list_estimate[i]/truth_value,\
            "normalized_on_estimate": on_policy_MC_agent.list_estimate[i]/truth_value,\
            "on_IS": on_policy_MC_agent.list_importance_sampling_ratio[i],\
            "on_IS_avg": on_policy_MC_agent.list_importance_sampling_ratio_avg[i],\
            "off_IS": ROA_agent.list_importance_sampling_ratio[i],\
            "off_IS_avg": ROA_agent.list_importance_sampling_ratio_avg[i],\
            "step_compare": i+1
            } )


print("done")        

# def