import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128,fc2_units = 128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.out = nn.Linear(fc2_units, action_size)

        self.dropout = nn.Dropout(p=0.2)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size , fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc1.weight.data.xavier_uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.xavier_uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.fc1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            # print(state.shape)
            state = torch.unsqueeze(state, 0)
        # print(state.shape)
        x = self.fc1(self.bn0(state))
        # print(x.shape)
        x = self.bn1(F.relu(x))
        x = self.bn2(F.relu(self.fc2(x)))

        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128,keep_prob=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # self.fcs1 = nn.Linear(state_size, fcs1_units)
        # self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.out = nn.Linear(fc3_units, 1)
        #self.dropout = nn.Dropout(p=keep_prob)
        self.dropout = nn.Dropout(p=keep_prob)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        #self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fcs1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # self.fcs1.weight.data.xavier_uniform_(*hidden_init(self.fcs1))
        # self.fc2.weight.data.xavier_uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.fcs1.bias.data.fill_(0.01)
        self.fc2.bias.data.fill_(0.01)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Random hack to avoid mismatch error
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        xs = self.bn1(F.relu(self.fcs1(state)))
        xs = torch.cat((xs, action.float()), dim=1)

        x = F.relu(self.fc2(xs))
        return self.fc3(x)

