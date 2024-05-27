import sys
import wandb
import yaml
import os
# import gym
import gymnasium as gym
import numpy as np
import torch

import pickle
from utilities.Config import Config
from utilities.Policy import *
from utilities.Torch_utils import *
from agents.MCAgent import *
from agents.OPEAgent import *

wandb_switch = True

if wandb_switch:
    with open('./02_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    env_id = wandb.config.env_id
    num_train_episode = wandb.config.num_train_episode
    a_repeat = wandb.config.a3_repeat
    # policy_index = wandb.config.policy_index
    policy_id = wandb.config.policy_id
else:
    env_id = "Hopper-v4"
    num_train_episode = 100
    a_repeat = 0
    # policy_index = 0
    policy_id = 0


config = Config()

with open(f'old_data_mujoco/02_ground_truth_value/PPO_7_env_50000_avg', 'rb') as fp:
# with open(f'data_mujoco/02_ground_truth_value/10_policy_avg', 'rb') as fp:
    truth_dic = pickle.load(fp)
truth_value = truth_dic[(env_id, policy_id, "reward")]
truth_step = truth_dic[(env_id, policy_id, "step")]

tag =  "|".join([str(env_id)] + [str(policy_id)])
# print("tag:", tag)
if wandb_switch:
    wandb.log({"tag": tag, "previous_truth_step": truth_step})

if truth_step < 200:
    # print(truth_step)
    config.device = "cpu"
    config.env = make_env(env_id)
    config.num_train_episode = num_train_episode

    ppo_agent = PPOMujocoAgent(config.env)
    ppo_agent.load_state_dict(torch.load(f"constructed_policy/{env_id}/ppo_continuous_action_{policy_id}.cleanrl_model", map_location=torch.device('cpu')))

    # print("----------------------------PPO1-------------------------")
    config.target_policy = ppo_agent
    config.behavior_policy = ppo_agent
    config.wandb_switch = wandb_switch
    config.is_truth_agent = True     #For truth agent
    config.eval_type = "ON"
    config.agent_type = None
    config.agent_name = "PPO1"
    truth_agent = TabularVMCAgent(config)
    truth_agent.run_all_episode()

print("done")
wandb.finish()



