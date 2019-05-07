from ddpg_agent import Agent,ReplayBuffer,OUNoise
from model import Actor, Critic
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
from utilities import soft_update, transpose_to_tensor, transpose_list
import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
HIDDEN_LAYERS=(512,256)
UPDATE_EVERY = 1
DROPOUT =0.2
NOISE_START = 1.0       # epsilon decay for the noise process added to the actions
NOISE_DECAY = 1e-6      # decay for for subrtaction of noise
NOISE_SIGMA = 0.2       # sigma for Ornstein-Uhlenbeck noise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG(object):
    """Multi agent class  that contains the two DDPG agents and a  shared replay buffer."""

    def __init__(self,state_size,action_size,num_agents,random_seed):
        """

        :param agent_tuple: {"state_size", "action_size", "random_seed","num_agents", "hidden_sizes"}
        :param evaluation_only:
        """
        #super(MADDPG, self).__init__()
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.action_size = action_size
        self.state_size = state_size * self.num_agents # since 24 is just for one
        self.agent_idx = np.arange(self.num_agents)
        # 2 agents
        # Actor Network (w/ Target Network)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.maddpg_agents = [Agent(state_size, action_size, random_seed, self) for _ in range(self.num_agents)]
        self.noise_weight = NOISE_START
        self.t_step = 0
        self.__name__ = 'MADDPG'

    # def __len__(self):
    #     return self.num_agents
    #
    # def __getitem__(self, key):
    #     return self.maddpg_agents[key]

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Split into agent wise tuples for memory
        experience = zip(self.maddpg_agents, states, actions, rewards, next_states,
                         dones)

        for i, exp in enumerate(experience):
            agent, state, action, reward, next_state, done = exp
        # # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #
        # if self.t_step == 0:
        #     # Learn, if enough samples are available in memory
        #     if len(self.memory) > BATCH_SIZE:
        #         for _ in range(num_updates):
        #             experiences = self.memory.sample()
        #             self.learn(experiences, GAMMA)
            player = self.agent_idx[self.agent_idx != i] # Choose the opposite player
            # Record the external player's states and actions separately for replay buffer
            ext_state = states[player]
            ext_action = actions[player]
            ext_next_state = next_states[player]
            agent.step(state, ext_state, action, ext_action, reward, next_state, ext_next_state, done)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        act_dim = np.zeros([self.num_agents, self.action_size])
        for player, agent in enumerate(self.maddpg_agents):
            act_dim[player,:] = agent.act(states[player],self.noise_weight,add_noise=True)
        self.noise_weight -= NOISE_DECAY
        return act_dim

    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

