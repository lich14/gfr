import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class TwinnedQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.Q1 = QNetwork(num_inputs, num_actions, hidden_dim)
        self.Q2 = QNetwork(num_inputs, num_actions, hidden_dim)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def act(self, states):
        action_logits = self.forward(states)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, return_probs=False):
        action_logits = self.forward(states)
        action_probs = torch.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        log_action_probs = None
        if return_probs:
            # Avoid numerical instability.
            z = (action_probs == 0.0).float() * 1e-8
            log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
