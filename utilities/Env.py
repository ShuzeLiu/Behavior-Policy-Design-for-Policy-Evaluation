
import gymnasium as gym
import numpy as np
import torch

class ClipAndDiscreteActionDimension0(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        division_number = 10
        interval = (self.action_space.high[0] - self.action_space.low[0]) / division_number
        action_on_d1 = action[0]
        # print(action, action_on_d1)
        action_on_d1 = min( torch.div((action_on_d1 - self.action_space.low[0]), interval, rounding_mode='trunc'), torch.tensor(division_number - 1))
        action_on_d1 = int(action_on_d1.item())
        action_on_d1 = action_on_d1 * interval + self.action_space.low[0] + interval / 2
        # print(action, action_on_d1)
        action[0] = action_on_d1
        return action


def make_env(env_id):
    # def thunk():

    env = gym.make(env_id)
    # env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ClipAndDiscreteActionDimension0(env)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, 0, 10))
    return env
