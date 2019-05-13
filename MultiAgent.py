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

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1
DROPOUT =0.2
NUM_UPDATES = 1

NOISE_SIGMA = 0.2       # Ornstein-Uhlenbeck noise parameter, volatility
NOISE_THETA = 0.15      # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
NOISE_START = 3.0       # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 200        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay


PER_ALPHA = 0.6         # importance sampling exponent
PER_BETA = 0.4          # prioritization exponentNOISE_SIGMA = 0.2       # sigma for Ornstein-Uhlenbeck noise
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Meta_AgentDDPG(object):
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
        self.state_size = state_size # since 24 is just for one
        self.agent_idx = np.arange(self.num_agents)
        # 2 agents
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, keep_prob=DROPOUT).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, keep_prob=DROPOUT).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # self.maddpg_agents = [Agent(state_size, action_size, random_seed, self) for _ in range(self.num_agents)]
        # #self.noise_weight = NOISE_START
        self.t_step = 0
        self.__name__ = 'MADDPG'

    def __len__(self):
        return self.num_agents

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.t_step += 1
        for i in range(self.num_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])


        # Learn every UPDATE_EVERY time steps.
        #self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step% UPDATE_EVERY == 0:
            for i in range(NUM_UPDATES):
                # Learn, if enough samples are available in memory
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
        # # update noise decay parameter
        # self.eps -= self.eps_decay
        # self.eps = max(self.eps, EPS_FINAL)
        # self.noise.reset()


    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    # def step(self, states, actions, rewards, next_states, dones):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     self.t_step +=1
    #     # Split into agent wise tuples for memory
    #     experience = zip(self.maddpg_agents, states, actions, rewards, next_states,
    #                      dones)
    #
    #     for i, exp in enumerate(experience):
    #         agent, state, action, reward, next_state, done = exp
    #
    #         agent.step(state, action, reward, next_state, done, i)



    # def act(self, states, add_noise=True):
    #     """Returns actions for given state as per current policy."""
    #     act_dim = np.zeros([self.num_agents, self.action_size])
    #     # ---- Test random actions initially for debugging
    #     # if self.t_step<BATCH_SIZE:
    #     #     actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
    #     #     act_dim = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    #     # else:
    #     for player, agent in enumerate(self.maddpg_agents):
    #         act_dim[player,:] = agent.act(states,add_noise=True)
    #         #self.noise_weight -= NOISE_DECAY
    #     return act_dim
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size)) # (2,2)
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            for agent_num, state in enumerate(states):
                actions[agent_num, :] = self.actor_local(state).cpu().data.numpy()
            #print(action)
        self.actor_local.train()
        # add noise to actions
        if add_noise:
            actions +=  self.noise.sample() # *self.eps
            #print("With noise \n",action)
        actions = np.clip(actions, -1, 1)
        return actions

    def reset(self):
        self.noise.reset()
        # for agent in self.maddpg_agents:
        #     agent.reset()

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
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
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
        self.eps = NOISE_START
        self.eps_decay = 1 / (EPS_EP_END)  # set decay rate based on epsilon end target

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size,  random_seed, keep_prob=DROPOUT).to(device)
        self.critic_target = Critic(state_size, action_size,  random_seed, keep_prob=DROPOUT).to(device)
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

    def step(self, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.t_step += 1
        self.memory.add(state, action, reward, next_state, done)


        # Learn every UPDATE_EVERY time steps.
        #self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step% UPDATE_EVERY == 0:
            for i in range(NUM_UPDATES):
                # Learn, if enough samples are available in memory
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA,agent_number)
        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()

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



    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number):
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
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        # Construct next actions vector relative to the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:, 2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:, :2], actions_next), dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # Construct action prediction vector relative to each agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:, 2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:, :2], actions_pred), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)



    # def learn(self, experiences, gamma):
    #     """Update policy and value parameters using given batch of experience tuples.
    #     Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    #     where:
    #         actor_target(state) -> action
    #         critic_target(state, action) -> Q-value
    #     Params
    #     ======
    #         experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
    #         gamma (float): discount factor
    #     """
    #     states, ext_states, actions, ext_actions, rewards, next_states, ext_next_states, dones = experiences
    #     full_states = torch.cat((states, ext_states), dim=1).to(device)
    #     full_actions = torch.cat((actions, ext_actions), dim=1).to(device)
    #     full_next_states = torch.cat((next_states, ext_next_states), dim=1).to(device)
    #
    #     # ---------------------------- update critic ---------------------------- #
    #     # Get predicted next-state actions and Q values from target models
    #     # Actions need to be calculated on the states individually and then concatenated
    #
    #     combined_actions_next = torch.cat((self.actor_target(states), self.actor_target(ext_states)), dim=1).to(device)
    #     # actions_next = self.actor_target(next_states)
    #     Q_targets_next = self.critic_target(full_next_states, combined_actions_next)
    #     # Compute Q targets for current states (y_i)
    #     Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #     # Compute critic loss
    #     Q_expected = self.critic_local(full_states, full_actions)
    #     critic_loss = F.mse_loss(Q_expected, Q_targets)
    #     # Minimize the loss
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
    #     self.critic_optimizer.step()
    #
    #     # ---------------------------- update actor ---------------------------- #
    #     # Compute actor loss
    #     actions_pred = torch.cat((self.actor_local(states), self.actor_local(ext_states).detach()), dim=1).to(device)
    #     actor_loss = -self.critic_local(full_states, actions_pred).mean()
    #     # Minimize the loss
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
    #
    #     # ----------------------- update target networks ----------------------- #
    #     self.soft_update(self.critic_local, self.critic_target, TAU)
    #     self.soft_update(self.actor_local, self.actor_target, TAU)
    #
    #     # update noise decay parameter
    #     self.eps -= self.eps_decay
    #     self.eps = max(self.eps, EPS_FINAL)
    #     self.noise.reset()

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
        self.size = size
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(0, 1, self.size)
        self.state = x + dx
        return self.state


class ModReplayBuffer:
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

def weighted_mse_loss(input, target, weights):
    '''
    Return the weighted mse loss to be used by Prioritized experience replay
    :param input: torch.Tensor.
    :param target: torch.Tensor.
    :param weights: torch.Tensor.
    :return loss:  torch.Tensor.
    '''
    # source: http://
    # forums.fast.ai/t/how-to-make-a-custom-loss-function-pytorch/9059/20
    out = (input-target)**2
    out = out * weights.expand_as(out)
    loss = out.mean(0)  # or sum over whatever dimensions
    return loss


class PrioritizedReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        '''Initialize a ReplayBuffer object.
        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size: int: size of each training batch
        :param seed: int: random seed
        :param alpha: float: 0~1 indicating how much prioritization is used
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state",
                                                  "action",
                                                  "reward",
                                                  "next_state",
                                                  "done"])
        self.seed = random.seed(seed)
        # specifics for prioritized replay
        self.alpha = max(0., alpha)  # alpha should be >= 0
        self.priorities = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self.cum_priorities = 0.
        self.eps = 1e-6
        self._indexes = []
        self.max_priority = 1.**self.alpha

    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        # exclude the value that will be discareded
        if len(self.priorities) >= self._buffer_size:
            self.cum_priorities -= self.priorities[0]
        # include the max priority possible initialy
        self.priorities.append(self.max_priority)  # already use alpha
        # accumulate the priorities abs(td_error)
        self.cum_priorities += self.priorities[-1]

    def sample(self):
        '''
        Sample a batch of experiences from memory according to importance-
        sampling weights
        :return. tuple[torch.Tensor]. Sample of past experiences
        '''
        i_len = len(self.memory)
        na_probs = None
        if self.cum_priorities:
            na_probs = np.array(self.priorities)/self.cum_priorities
        l_index = np.random.choice(i_len,
                                   size=min(i_len, self.batch_size),
                                   p=na_probs)
        self._indexes = l_index

        experiences = [self.memory[ii] for ii in l_index]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def _calculate_is_w(self, f_priority, current_beta, max_weight, i_n):
        #  wi= ((N x P(i)) ^ -β)/max(wi)
        f_wi = (i_n * f_priority/self.cum_priorities)
        return (f_wi ** -current_beta)/max_weight

    def get_is_weights(self, current_beta):
        '''
        Return the importance sampling (IS) weights of the current sample based
        on the beta passed
        :param current_beta: float. fully compensates for the non-uniform
            probabilities P(i) if β = 1
        '''
        # calculate P(i) to what metters
        i_n = len(self.memory)
        max_weight = (i_n * min(self.priorities) / self.cum_priorities)
        max_weight = max_weight ** -current_beta

        this_weights = [self._calculate_is_w(self.priorities[ii],current_beta,max_weight,i_n)
                        for ii in self._indexes]
        return torch.tensor(this_weights,device=device,dtype=torch.float).reshape(-1, 1)

    def update_priorities(self, td_errors):
        '''
        Update priorities of sampled transitions
        inspiration: https://bit.ly/2PdNwU9
        :param td_errors: tuple of torch.tensors. TD-Errors of last samples
        '''
        for i, f_tderr in zip(self._indexes, td_errors):
            f_tderr = float(f_tderr)
            self.cum_priorities -= self.priorities[i]
            # transition priority: pi^α = (|δi| + ε)^α
            self.priorities[i] = ((abs(f_tderr) + self.eps) ** self.alpha)
            self.cum_priorities += self.priorities[i]
        self.max_priority = max(self.priorities)
        self._indexes = []

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)

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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)