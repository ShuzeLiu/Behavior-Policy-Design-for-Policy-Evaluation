import sys
import wandb
import yaml
import gymnasium as gym
import time
import pickle
from utilities.Config import Config
from utilities.Policy import *
from utilities.Torch_utils import *
from agents.MCAgent import *
from agents.OPEAgent import *
from environments.GridWorld import GridWorld

wandb_switch = True


if wandb_switch:
    with open('./02_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    env_size = wandb.config.env_size
    policy_id = wandb.config.policy_id
    num_train_episode = wandb.config.num_train_episode
    evaluation_method = wandb.config.evaluation_method

else:
    env_size = 5
    policy_id = 30
    num_train_episode = 10
    evaluation_method = "on"


with open('data/01_policy_data', 'rb') as fp:
    list_poilcy = pickle.load(fp)
target_policy = list_poilcy[policy_id]

env = GridWorld(width = env_size,  \
    t_max = env_size, \
    offline_data_number = 1
    )



config = Config()
config.env = env
config.num_train_episode = num_train_episode
config.target_policy = target_policy

if evaluation_method == "on":
    config.behavior_policy = target_policy
else: 
    config.behavior_policy = list_poilcy[0]

config.wandb_switch = wandb_switch
config.is_truth_agent = True     #For truth agent
config.device = "cpu"
truth_agent = TabularVMCAgent(config)
truth_agent.run_all_episode()

print("done")



wandb.finish()



