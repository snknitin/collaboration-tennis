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
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.993            # discount factor
TAU = 8e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 10
DROPOUT =0.2
NUM_UPDATES = 5
MAX_T = 1000

NOISE_SIGMA = 0.2       # Ornstein-Uhlenbeck noise parameter, volatility
NOISE_THETA = 0.15      # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
NOISE_START = 5.0       # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay


PER_ALPHA = 0.6         # importance sampling exponent
PER_BETA = 0.4          # prioritization exponent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        super(MADDPG_Agent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.__name__ = 'MADDPG'
        self.seed = random.seed(random_seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.alpha = PER_ALPHA
        self.initial_beta = PER_BETA
        self.max_t = MAX_T

        self.eps = NOISE_START
        self.eps_decay = 1/(EPS_EP_END*NUM_UPDATES)  # set decay rate based on epsilon end target
        self.timestep = 0

        # Actor Network (w/ Target Network)
        # No modification of action size with respect to number of agents
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
        self.noise = OUNoise((self.num_agents, self.action_size), random_seed)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed,self.alpha)

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_beta(self, t):
        '''
        Return the current exponent β based on its schedule. Linearly anneal β
        from its initial value β0 to 1, at the end of learning.
        :param t: integer. Current time step in the episode
        :return current_beta: float. Current exponent beta
        '''
        f_frac = min(float(t) / self.max_t, 1.0)
        current_beta = self.initial_beta + f_frac * (1. - self.initial_beta)
        return current_beta


    def step(self, state, action, reward, next_state, done, agent_number):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.timestep += 1
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        # Learn, if enough samples are available in memory and at learning interval settings
        if len(self.memory) > BATCH_SIZE and self.timestep % UPDATE_EVERY == 0:
            for _ in range(NUM_UPDATES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_number)

    def act(self, states, add_noise=True):
        """Returns actions for both agents as per current policy, given their respective states."""
        states = torch.from_numpy(states).float().to(device)
        #actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            #for agent_num, state in enumerate(states):
            action = self.actor_local(states).cpu().data.numpy()
                #actions[agent_num, :] = action
        self.actor_local.train()
        # add noise to actions
        if add_noise:
            action += self.eps * self.noise.sample()
        actions = np.clip(action, -1, 1)
        return actions

    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma, agent_number,t=1000):
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
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        # Compute Q targets for current states (y_i)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        # compute importance-sampling weight wj
        f_currbeta = self.get_beta(t)
        weights = self.memory.get_is_weights(current_beta=f_currbeta)

        # compute TD-error δj and update transition priority pj
        td_errors = Q_targets - Q_expected
        self.memory.update_priorities(td_errors)


        critic_loss = weighted_mse_loss(Q_expected, Q_targets,weights)
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
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # update noise decay parameter
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()

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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        # Shape here is (2,2) because the env has 2 agents' actions
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.scale = 0.1
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