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
LR_ACTOR = 2e-3         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
HIDDEN_LAYERS=(512,256)
UPDATE_EVERY = 16
DROPOUT =0.2
NUM_UPDATES = 2
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

    def __len__(self):
        return self.num_agents

    def __getitem__(self, key):
        return self.maddpg_agents[key]

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
        #self.noise_weight -= NOISE_DECAY
        return act_dim

    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()


class Agent(object):
    """
    Each actor takes a state input for a single agent while the critic takes
    a concatentation of the states and actions from all agents.
    """

    def __init__(self, state_size, action_size, random_seed, maddpg):
        """Initialize an Double Agent object.

        Params
        ======
            id (str) : Player 1 or player 2 (possibly 0 and 1)
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            maddpg : The multi agent
        """
        # super(Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = maddpg.num_agents
        self.__name__ = 'DDPG'

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, self.num_agents, random_seed, keep_prob=DROPOUT).to(device)
        self.critic_target = Critic(state_size, action_size, self.num_agents, random_seed, keep_prob=DROPOUT).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = maddpg.memory
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state, ext_state, action, ext_action, reward, next_state, ext_next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, ext_state, action, ext_action, reward, next_state, ext_next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            for i in range(NUM_UPDATES):
                # Learn, if enough samples are available in memory
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    # def step2(self, states, actions, rewards, next_states, dones,num_update = 1):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     # Save experience / reward
    #     for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
    #         self.memory.add(state, action, reward, next_state, done)
    #
    #     # Learn every UPDATE_EVERY time steps.
    #     self.t_step = (self.t_step + 1) % UPDATE_EVERY
    #     if self.t_step == 0:
    #         for i in range(num_update):
    #             # Learn, if enough samples are available in memory
    #             if len(self.memory) > BATCH_SIZE:
    #                 experiences = self.memory.sample()
    #                 self.learn(experiences, GAMMA)

    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, ext_states, actions, ext_actions, rewards, next_states, ext_next_states, dones = experiences
        full_states = torch.cat((states, ext_states), dim=1).to(device)
        full_actions = torch.cat((actions, ext_actions), dim=1).to(device)
        full_next_states = torch.cat((next_states, ext_next_states), dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Actions need to be calculated on the states individually and then concatenated

        combined_actions_next = torch.cat((self.actor_target(states), self.actor_target(ext_states)), dim=1).to(device)
        # actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(full_next_states, combined_actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = torch.cat((self.actor_local(states), self.actor_local(ext_states).detach()), dim=1).to(device)
        actor_loss = -self.critic_local(full_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "ext_state", "action", "ext_action",
                                                                "reward", "next_state", "ext_next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, ext_state, action, ext_action, reward, next_state, ext_next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, ext_state, action, ext_action, reward, next_state, ext_next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        ext_states = torch.from_numpy(np.vstack([e.ext_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        ext_actions = torch.from_numpy(np.vstack([e.ext_action for e in experiences if e is not None])).float().to(
            device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        ext_next_states = torch.from_numpy(
            np.vstack([e.ext_next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, ext_states, actions, ext_actions, rewards, next_states, ext_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)