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
from agents import *


#Used to debug
wandb_switch = True


if wandb_switch:
    with open('./03_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)

    wandb_env_size = wandb.config.a_env_size
    wandb_policy_id = wandb.config.policy_id
    wandb_offline_data_number =  wandb.config.offline_data_number
    wandb_learning_rate_w_q = 0.001 * pow(2, wandb.config.learning_rate_w_q)
    wandb_learning_rate_w_1 = 0.001 * pow(2, wandb.config.learning_rate_w_1)
    wandb_learning_rate_w_2 = 0.001 * pow(2, wandb.config.learning_rate_w_2)
    wandb_learning_rate_w_r = 0.001 * pow(2, wandb.config.learning_rate_w_r)
    wandb_learning_rate_w_u = 0.001 * pow(2, wandb.config.learning_rate_w_u)
    wandb_feature_type = wandb.config.feature_type
    wandb_num_epoch = wandb.config.num_epoch
    wandb_repeat = wandb.config.a_repeat
    wandb_policy_id =  wandb.config.policy_id
    num_train_episode = 50
    wandb_layer_structure = wandb.config.layer_structure
    wandb_train_or_load =  wandb.config.train_or_load
    wandb_global_repeat = wandb.config.aaa_global_repeat

else:
    wandb_env_size = 30
    wandb_offline_data_number = 100
    wandb_learning_rate_w_q = pow(2, -2)
    wandb_learning_rate_w_1 = pow(2, 2)
    wandb_learning_rate_w_2 = pow(2, 2)
    wandb_learning_rate_w_r = pow(2, -2)
    wandb_learning_rate_w_u = pow(2, -2)
    wandb_feature_type = "pseudo_tabular"
    wandb_num_epoch = 10
    num_train_episode = 3
    wandb_repeat = 1
    wandb_policy_id = 1
    wandb_layer_structure = "2"
    wandb_train_or_load = "load"
    wandb_global_repeat = 1

if wandb_feature_type == "pseudo_tabular":
    if wandb_env_size == 5: wandb_layer_structure =  "2"
    if wandb_env_size == 10: wandb_layer_structure = "2"
    if wandb_env_size == 15: wandb_layer_structure = "1"
    if wandb_env_size == 20: wandb_layer_structure = "2"
    if wandb_env_size == 30: wandb_layer_structure = "1"
    if wandb_env_size == 50: wandb_layer_structure = "0.015"
    if wandb_env_size == 60: wandb_layer_structure = "0.015"
    if wandb_env_size == 90: wandb_layer_structure = "0.015"
    if wandb_env_size == 100: wandb_layer_structure = "0.015"
elif wandb_feature_type == "tabular":
    wandb_layer_structure = ""

tag =  "|".join(["ODI", str(wandb_env_size), wandb_layer_structure, str(wandb_global_repeat)])
print("tag:", tag)
if wandb_switch:
    wandb.log({"tag": tag})
    

env = environments.GridWorld(width = wandb_env_size,  \
    t_max = wandb_env_size, \
    offline_data_number = wandb_offline_data_number,\
    feature_type = wandb_feature_type
    )
with open('data/01_policy_data', 'rb') as fp:
    list_poilcy = pickle.load(fp)
target_policy = list_poilcy[wandb_policy_id]

if wandb_env_size >= 60:
    with open('data/large_linear/02_ground_truth', 'rb') as fp:
        dic_truth = pickle.load(fp)
else:
    with open('data/02_ground_truth', 'rb') as fp:
        dic_truth = pickle.load(fp)
truth_value = dic_truth[(wandb_env_size, wandb_policy_id)]

print(truth_value)

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
config_OPE.target_policy = target_policy
config_OPE.is_truth_agent = False
config_OPE.truth_value = truth_value


config_OPE.wandb_switch = wandb_switch
config_OPE.batch_size = 128
config_OPE.num_epoch = wandb_num_epoch
# config_OPE.num_epoch = 2
config_OPE.training_percent = 0.7
config_OPE.num_epoch_w_q = config_OPE.num_epoch
config_OPE.learning_rate_w_q = wandb_learning_rate_w_q
config_OPE.num_epoch_w_1 = config_OPE.num_epoch
config_OPE.learning_rate_w_1 = wandb_learning_rate_w_1
config_OPE.num_epoch_w_2 = config_OPE.num_epoch
config_OPE.learning_rate_w_2 = wandb_learning_rate_w_2
config_OPE.num_epoch_w_r = config_OPE.num_epoch
config_OPE.learning_rate_w_r = wandb_learning_rate_w_r
config_OPE.num_epoch_w_u = config_OPE.num_epoch
config_OPE.learning_rate_w_u = wandb_learning_rate_w_u


if wandb_train_or_load == "train":
    OPE_agent = agents.OPEAgent(config_OPE)
    config_OPE.behavior_policy = OPE_agent.ope_policy
elif wandb_train_or_load == "load":
    name = "e_"+ str(config_OPE.env.width) + "_re_" + str(config_OPE.repeat)\
               + "_pid_" + str(config_OPE.policy_id) + "_" + str(config_OPE.feature_type)
    if wandb_env_size >= 60:
        model_w_u = torch.load("data/large_linear/03_trained_model_linear/"+name, map_location=config_OPE.device)
    else:
        model_w_u = torch.load("data/03_trained_model_linear1/"+name, map_location=config_OPE.device)
    model_w_u.eval()
    config_OPE.behavior_policy = OPEPolicy(env, target_policy, model_w_u, config_OPE.device)
    

OPE_MC_agent = agents.TabularVMCAgent(config_OPE)
OPE_MC_agent.run_all_episode()


config_on_MC = utilities.Config()
config_on_MC.wandb_switch = wandb_switch
config_on_MC.device = 'cpu'
config_on_MC.env = env
config_on_MC.num_train_episode = num_train_episode
config_on_MC.target_policy = target_policy
config_on_MC.behavior_policy = target_policy
config_on_MC.is_truth_agent = False
config_on_MC.truth_value = truth_value

on_policy_MC_agent = agents.TabularVMCAgent(config_on_MC)
on_policy_MC_agent.run_all_episode()



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
    
    if wandb_switch:
    # if i == num_train_episode - 1 and wandb_switch:
        wandb.log({"normalized_OPE_error": OPE_error/truth_value,   \
            "normalized_on_error": on_error/truth_value, \
            "acc_normalized_error_diff":  acc_on_error/truth_value -  acc_OPE_error/truth_value, \
            "normalized_error_diff":  on_error/truth_value -  OPE_error/truth_value, \
            "var_ratio": var_OPE/var_on, \
            "normalized_OPE_estimate": OPE_MC_agent.list_estimate[i]/truth_value,\
            "normalized_on_estimate": on_policy_MC_agent.list_estimate[i]/truth_value,\
            # "on_IS": on_policy_MC_agent.list_importance_sampling_ratio[i],\
            "off_IS": OPE_MC_agent.list_importance_sampling_ratio[i],\
            "step_compare": i+1
            } )

if wandb_switch:
    wandb.finish()



