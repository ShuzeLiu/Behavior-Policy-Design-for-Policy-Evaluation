import torch
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
        if config.feature_type == "pseudo_tabular":
            neuron_number_list = [config.env.num_states] + [int(float(config.wandb_layer_structure)*config.env.num_states)] + [config.env.num_actions]
            #mutilple layer
            # neuron_number_list = [config.env.num_states] + [int(float(c)*config.env.num_states) for c in config.wandb_layer_structure] + [config.env.num_actions]
            # print(neuron_number_list)
            # for i in range(len(config.wandb_layer_structure)):
            for i in range(1):
                layers.append(nn.Linear(neuron_number_list[i], neuron_number_list[i+1], bias=True))
                # nn.init.constant_(layers[-1].weight.data, 0)
                # print(layers[-1].weight.data.shape)
                layers.append(nn.ReLU())
            layers.append(nn.Linear(neuron_number_list[-2], neuron_number_list[-1], bias=True))
        elif config.feature_type == "tabular":
            layers.append(nn.Linear(config.env.num_states, config.env.num_actions, bias=False))

        # nn.init.constant_(layers[-1].weight.data, 0)
        self.nn_stack = nn.Sequential(*layers)

        self.to(config.device)

    def forward(self, x):
        logits = self.nn_stack(x)
        return logits


class GridActionNetwork(nn.Module):
    def __init__(self, config, dis):
        super().__init__()

        hidden_layer_size = [int(i) for i in config.wandb_layer_structure.split("|")]
        layers = []


        for i,s in enumerate(hidden_layer_size):
            if i == 0:
                layers.append(nn.Linear(config.env.num_states, hidden_layer_size[i], bias=True))
            else:
                layers.append(nn.Linear(hidden_layer_size[i-1], hidden_layer_size[i], bias=True))
            layers[-1].weight.data = torch.rand(layers[-1].weight.data.shape)-1
            layers[-1].bias.data = torch.rand(layers[-1].bias.data.shape)-1
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_layer_size[-1], config.env.num_actions, bias=True))
        # layers[-1].weight.data = torch.zeros(layers[-1].weight.data.shape)
        # layers[-1].weight.requires_grad  = False
        layers[-1].bias.data = torch.log(torch.tensor(dis))

        layers.append(nn.Softmax(dim = 0))

        self.nn_stack = nn.Sequential(*layers)
        self.to(config.device)

    def forward(self, x):
        logits = self.nn_stack(x)
        return logits
