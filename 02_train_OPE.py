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

#Todo:
#1. check to remove termination state
#2. sample data from random policy
#3. in the data_format add the unchanged action to the observation
#4. check if reward is float32
#5. change the data counter
#7. when the reward is negative add abs on the compare step
#Used to debug

wandb_switch = True
wandb_sweep_id = "tune target network frequency"
# wandb_sweep_id = "test"


if wandb_switch:
    with open('./03_config_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    run = wandb.init(config=config)
    wandb_env_id = wandb.config.env_id
    wandb_policy_id = wandb.config.policy_id
    wandb_offline_data_number =  wandb.config.offline_data_number
    wandb_learning_rate_w_q = 0.001 * pow(2, wandb.config.learning_rate_w_q)
    wandb_learning_rate_w_r = 0.001 * pow(2, wandb.config.learning_rate_w_r)
    wandb_learning_rate_w_u = 0.001 * pow(2, wandb.config.learning_rate_w_u)
    wandb_num_epoch = wandb.config.num_epoch
    wandb_repeat = wandb.config.a_repeat
    num_train_episode = 200
    wandb_layer_structure = wandb.config.z_layer_structure
    wandb_train_or_load =  wandb.config.train_or_load
    wandb_target_network_frequency =   wandb.config.target_network_frequency
    wandb_data_generation = wandb.config.data_generation
    wandb_max_epsilon = wandb.config.max_epsilon
    # wandb_policy_index = wandb.config.policy_index
    wandb_inner_repeat = 100
    wandb_time_feature = wandb.config.time_feature
else:
    # wandb_env_id = "InvertedPendulum-v4"
    wandb_env_id = "Hopper-v4"
    wandb_policy_id = 1
    wandb_offline_data_number = 20
    wandb_learning_rate_w_q = pow(2, -2)
    wandb_learning_rate_w_r = pow(2, -2)
    wandb_learning_rate_w_u = pow(2, -2)
    wandb_num_epoch = 2
    num_train_episode = 10
    wandb_repeat = 1
    wandb_layer_structure = [64,64]
    wandb_train_or_load = "train"
    wandb_target_network_frequency = 10
    wandb_division_number = 10
    wandb_data_generation = "off"
    wandb_max_epsilon = 0.1
    # wandb_policy_index = 0
    wandb_inner_repeat = 5 
    wandb_time_feature = 1
# with open("./old_data_mujoco/03_trained_OPE_model_7_env_100_policy_dic/02_working_policy_id_dic", 'rb') as fp:
#     working_policy_id_dic = pickle.load(fp)
# wandb_policy_id = working_policy_id_dic[wandb_env_id][wandb_policy_index]
# print("wandb_policy_id:", wandb_policy_id)

tag =  "|".join(\
    # ["debug_off_number"]+\ 
                [wandb_env_id[:6]] + ["pid_" + str(wandb_policy_id)] + ["re_" + str(wandb_repeat)] +  [str(i) for i in wandb_layer_structure] + ["data_"+ str(wandb_offline_data_number)] + ["epoch_" + str(wandb_num_epoch)] + ["target_network_frequency" + str(wandb_target_network_frequency)]  + ["gener_" + wandb_data_generation] + ["max_eps_" + str(wandb_max_epsilon)]+ ["wandb_switch_" + str(wandb_switch)] + ["sweep_" + str(wandb_sweep_id)] + ["time_feature_" + str(wandb_time_feature)] )
print("tag:", tag)
if wandb_switch:
    wandb.log({"tag": tag})

with open(f'data_mujoco/02_ground_truth_value/5_env_100_policy_avg', 'rb') as fp:
    truth_dic = pickle.load(fp)

if (wandb_env_id, wandb_policy_id, "reward") in truth_dic:
    truth_value = truth_dic[(wandb_env_id, wandb_policy_id, "reward")]
    truth_step = truth_dic[(wandb_env_id, wandb_policy_id, "step")]
    print(f"({wandb_env_id}, {wandb_policy_id}, reward)", truth_value)
    print(f"({wandb_env_id}, {wandb_policy_id}, step)", truth_step)
    if wandb_switch:
        wandb.log({"truth_value": truth_value, "truth_step": truth_step})

    env = make_env(wandb_env_id)
    ppo_agent = PPOMujocoAgent(env)
    # ppo_agent.load_state_dict(torch.load(f"constructed_policy/InvertedPendulum-v4/ppo_continuous_action_{wandb_policy_id}.cleanrl_model", map_location=torch.device('cpu')))
    ppo_agent.load_state_dict(torch.load(f"constructed_policy/{wandb_env_id}/ppo_continuous_action_{wandb_policy_id}.cleanrl_model", map_location=torch.device('cpu')))
    off_agent = OffMujocoAgent(env, ppo_agent)
    off_agent_noise = OffMujocoAgentNoise(env, ppo_agent, 0.1)
    # termination_state = np.append(np.zeros(env.observation_space.shape),[0,1]) 
    def generate_data(env, data_number):
        if wandb_data_generation == "off":
            off_agent_pool = [off_agent_noise]
        else:
            off_agent_pool = [off_agent]
        data = []
        i = 0
        for data_generate_policy in off_agent_pool:
            for _ in range(data_number):
                if wandb_data_generation == "off":
                    data_generate_policy.reset_epsilon()
                behavior_policy = data_generate_policy
                # behavior_policy = off_agent
                init_state, _ = env.reset()
                state = init_state
                done = False
                t = 0
                # while not done and len(data) < data_number:
                while not done:
                    t += 1
                    action, action_index = behavior_policy.get_action_and_index(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    done_indicator = 1 if done else 0
                    if wandb_time_feature:
                        data.append([np.append([t], state), (action, action_index), reward, np.append([t+1], next_state), done_indicator])
                    else:
                        data.append([state, (action, action_index), reward, next_state, done_indicator])
                    state = next_state

        if wandb_switch:
            wandb.log({"step_len_offline_data": len(data)})                      
        return data


    offline_data = generate_data(env, wandb_offline_data_number)


    target_policy = off_agent


    config_OPE = utilities.Config()
    config_OPE.time_feature = wandb_time_feature
    config_OPE.target_network_frequency = wandb_target_network_frequency
    config_OPE.eval_type = "OFF"
    config_OPE.agent_type = "discretization"
    config_OPE.agent_name = "PPO_discrete"
    config_OPE.wandb_layer_structure = wandb_layer_structure
    config_OPE.device = 'cpu'
    # config_OPE.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # config_OPE.device = torch.device('cuda:'+str(best_gpu()) if torch.cuda.is_available() else 'cpu')
    config_OPE.offline_data = offline_data
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
    config_OPE.batch_size = 512
    config_OPE.num_epoch = wandb_num_epoch
    # config_OPE.num_epoch = 2
    config_OPE.training_percent = 0.99
    config_OPE.num_epoch_w_q = config_OPE.num_epoch
    config_OPE.learning_rate_w_q = wandb_learning_rate_w_q
    config_OPE.num_epoch_w_r = config_OPE.num_epoch // 2
    config_OPE.learning_rate_w_r = wandb_learning_rate_w_r
    config_OPE.num_epoch_w_u = config_OPE.num_epoch
    config_OPE.learning_rate_w_u = wandb_learning_rate_w_u
    config_OPE.tag = tag
    config_OPE.wandb_sweep_id = wandb_sweep_id

    if wandb_train_or_load == "train":
        OPE_agent = agents.OPEAgent(config_OPE)
        model_w_u = OPE_agent.model_w_u
    elif wandb_train_or_load == "load":
        name = "env_"+ str(wandb_env_id) + "_re_" + str(config_OPE.repeat)\
                    + "_pid_" + str(config_OPE.policy_id) + "_" + "_".join([str(i)  for i in config_OPE.wandb_layer_structure])
        # name = "e_"+ str(config_OPE.env.width) + "_re_" + str(config_OPE.repeat)\
        #            + "_pid_" + str(config_OPE.policy_id) + "_" + str(config_OPE.feature_type)
        model_w_u = torch.load("data/cart_pole/03_trained_OPE_model_0/"+name, map_location=config_OPE.device)

    model_w_u.eval()
    off_agent.model_w_u = model_w_u


    agg_dic = collections.defaultdict(list)

    for _ in range(wandb_inner_repeat):
        OPE_MC_agent = agents.TabularVMCAgent(config_OPE)
        OPE_MC_agent.run_all_episode()
        agg_dic["OPE_list_v"].append(OPE_MC_agent.list_v)
        agg_dic["OPE_list_estimate"].append(OPE_MC_agent.list_estimate)
        agg_dic["OPE_list_error"].append(OPE_MC_agent.list_error)
        agg_dic["OPE_list_step"].append(OPE_MC_agent.list_step)





    config_on_MC = utilities.Config()
    config_on_MC.time_feature = wandb_time_feature
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

    for _ in range(wandb_inner_repeat):
        on_policy_MC_agent = agents.TabularVMCAgent(config_on_MC)
        on_policy_MC_agent.run_all_episode()
        agg_dic["on_list_v"].append(on_policy_MC_agent.list_v)
        agg_dic["on_list_estimate"].append(on_policy_MC_agent.list_estimate)
        agg_dic["on_list_error"].append(on_policy_MC_agent.list_error)
        agg_dic["on_list_step"].append(on_policy_MC_agent.list_step)

    # for k,v in agg_dic.items():
    #     print(k, len(v))
    #     print(k, len(v[0]))

    agg_dic["truth_value"] = truth_value
    agg_dic["truth_step"] = truth_step
    agg_dic["OPE_list_error_mean"] = np.mean(agg_dic["OPE_list_error"], axis = 0)
    agg_dic["on_list_error_mean"] = np.mean(agg_dic["on_list_error"], axis = 0)
    agg_dic["error_raito"] = agg_dic["on_list_error_mean"]/agg_dic["OPE_list_error_mean"]


    display_dic = {"on_OPE_ratio_0": agg_dic["error_raito"][0],\
            f"on_OPE_ratio_{(num_train_episode - 1)//2}": agg_dic["error_raito"][(num_train_episode - 1)//2],\
            f"on_OPE_ratio_{num_train_episode - 1}": agg_dic["error_raito"][num_train_episode - 1]}


    # print(display_dic)

    if wandb_switch:
        wandb.log({"on_OPE_ratio_0": agg_dic["error_raito"][0],\
            f"on_OPE_ratio_{(num_train_episode - 1)//2}": agg_dic["error_raito"][(num_train_episode - 1)//2],\
            f"on_OPE_ratio_{num_train_episode - 1}": agg_dic["error_raito"][num_train_episode - 1]}
            )


    name = config_OPE.tag
    # print(name)
    p = f"data_mujoco/03_trained_OPE_model_100_repeat_dic/{config_OPE.env_id}/"
    #check if p exists, otherwise create p
    if not os.path.exists(p):
        os.makedirs(p)


    with open(p + name, 'wb') as fp:
        pickle.dump(agg_dic, fp)


    print("compare end")

if wandb_switch:
    wandb.finish()



