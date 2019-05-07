from ddpg_agent import Agent,ReplayBuffer,OUNoise
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
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
HIDDEN_LAYERS=(512,256)
UPDATE_EVERY = 20
DROPOUT =0.2
NOISE_START = 1.0       # epsilon decay for the noise process added to the actions
NOISE_DECAY = 1e-6      # decay for for subrtaction of noise
NOISE_SIGMA = 0.2       # sigma for Ornstein-Uhlenbeck noise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MADDPG():
    """Multi agent class  that contains the two DDPG agents and a  shared replay buffer."""

    def __init__(self,agent_tuple1,agent_tuple2,action_size,num_agents):
        """

        :param agent_tuple: {"state_size", "action_size", "random_seed","num_agents", "hidden_sizes"}
        :param evaluation_only:
        """
        super(MADDPG, self).__init__()
        self.num_agents = num_agents
        self.seed = random.seed(42)
        # 2 agents
        self.maddpg_agent = [Agent(**agent_tuple1),
                             Agent(**agent_tuple2)]
        # Shared replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        self.noise_weight = NOISE_START


    def step(self, states, actions, rewards, next_states, dones, num_update = 1):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        states = states.reshape(1, -1)  # 2x24 into 1x48
        next_states = next_states.reshape(1, -1)  # 2x24 into 1x48
        self.memory.add(states, actions, rewards, next_states, dones)
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(num_update):
                    # each agent does it's own sampling from the replay buffer
                    experiences = [self.memory.sample() for _ in range(self.num_agents)]
                    self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        # pass each agent's state from the environment and calculate it's action
        actions = []
        for agent, state in zip(self.maddpg_agent, states):
            action = agent.act(state, self.noise_weight, add_noise=True)
            self.noise_weight -= NOISE_DECAY
            actions.append(action)
        return np.array(actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma):
        """
         Each agent uses it's own actor to calculate next_actions
        :param experiences:
        :Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """

        next_actions = []
        for i, agent in enumerate(self.maddpg_agent):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            next_actions.append(next_action)
        # each agent uses it's own actor to calculate actions
        actions = []
        for i, agent in enumerate(self.maddpg_agent):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            actions.append(action)
        # each agent learns from it's experience sample
        for i, agent in enumerate(self.maddpg_agent):
            agent.learn(i, experiences[i], gamma, next_actions, actions)

    # def get_actors(self):
    #     """get actors of all the agents in the MADDPG object"""
    #     actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
    #     return actors
    #
    # def get_target_actors(self):
    #     """get target_actors of all the agents in the MADDPG object"""
    #     target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
    #     return target_actors
    #
    #
    #
    # def update(self, samples, agent_number, logger):
    #     """update the critics and actors of all the agents """
    #
    #     # need to transpose each element of the samples
    #     # to flip obs[parallel_agent][agent_number] to
    #     # obs[agent_number][parallel_agent]
    #     obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
    #
    #     obs_full = torch.stack(obs_full)
    #     next_obs_full = torch.stack(next_obs_full)
    #
    #     agent = self.maddpg_agent[agent_number]
    #     agent.critic_optimizer.zero_grad()
    #
    #     # critic loss = batch mean of (y- Q(s,a) from target network)^2
    #     # y = reward of this timestep + discount * Q(st+1,at+1) from target network
    #     target_actions = self.target_act(next_obs)
    #     target_actions = torch.cat(target_actions, dim=1)
    #
    #     target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
    #
    #     with torch.no_grad():
    #         q_next = agent.critic_target(target_critic_input)
    #
    #     y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
    #     action = torch.cat(action, dim=1)
    #     critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
    #     q = agent.critic_local(critic_input)
    #
    #     huber_loss = torch.nn.SmoothL1Loss()
    #     critic_loss = huber_loss(q, y.detach())
    #     critic_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
    #     agent.critic_optimizer.step()
    #
    #     # update actor network using policy gradient
    #     agent.actor_optimizer.zero_grad()
    #     # make input to agent
    #     # detach the other agents to save computation
    #     # saves some time for computing derivative
    #     q_input = [self.maddpg_agent[i].actor_local(ob) if i == agent_number \
    #                    else self.maddpg_agent[i].actor_local(ob).detach()
    #                for i, ob in enumerate(obs)]
    #
    #     q_input = torch.cat(q_input, dim=1)
    #     # combine all the actions and observations for input to critic
    #     # many of the obs are redundant, and obs[1] contains all useful information already
    #     q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
    #
    #     # get the policy gradient
    #     actor_loss = -agent.critic_local(q_input2).mean()
    #     actor_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
    #     agent.actor_optimizer.step()
    #
    #     al = actor_loss.cpu().detach().item()
    #     cl = critic_loss.cpu().detach().item()
    #     logger.add_scalars('agent%i/losses' % agent_number,
    #                        {'critic loss': cl,
    #                         'actor_loss': al},
    #                        self.iter)
    #
    # def update_targets(self):
    #     """soft update targets"""
    #     self.iter += 1
    #     for ddpg_agent in self.maddpg_agent:
    #         soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, self.tau)
    #         soft_update(ddpg_agent.critic_target, ddpg_agent.actor_local, self.tau)