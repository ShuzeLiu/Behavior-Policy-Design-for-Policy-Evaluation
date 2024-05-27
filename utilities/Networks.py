import torch
import numpy as np
from torch import nn



class SingleLinearNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nn_stack = nn.Linear(config.env.num_states, 1, bias=False)
        self.to(config.device)

    def forward(self, x):
        logits = self.nn_stack(x)
        return logits


class OPENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # print("config.wandb_layer_structure:", config.wandb_layer_structure)
        layers = []
        neuron_number_list = [config.env.observation_space.shape[0] + config.env.action_space.shape[0] - 1 + config.time_feature] + config.wandb_layer_structure + [10]
        # print(neuron_number_list)
        #mutilple layer
        # neuron_number_list = [config.env.num_states] + [int(float(c)*config.env.num_states) for c in config.wandb_layer_structure] + [config.env.num_actions]
        # print(neuron_number_list)
        # for i in range(len(config.wandb_layer_structure)):
        for i in range(len(neuron_number_list)-2):
            layers.append(nn.Linear(neuron_number_list[i], neuron_number_list[i+1], bias=True))
            # nn.init.constant_(layers[-1].weight.data, 0)
            # print(layers[-1].weight.data.shape)
            layers.append(nn.ReLU())
        layers.append(nn.Linear(neuron_number_list[-2], neuron_number_list[-1], bias=True))

        # nn.init.constant_(layers[-1].weight.data, 0)
        self.nn_stack = nn.Sequential(*layers)
        self.to(config.device)

    def forward(self, x):
        logits = self.nn_stack(x)
        return logits


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)