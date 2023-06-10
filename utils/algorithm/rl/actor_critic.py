
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class A2CPolicyNet(nn.Module):
    """Implement both actor and critic in one model."""
    def __init__(self, obs_space_dim, action_space_dim, hidden_dim=128):
        super(A2CPolicyNet, self).__init__()
        self.affine1 = nn.Linear(obs_space_dim, hidden_dim)
        # actor's layer
        self.action_layer = nn.Linear(hidden_dim, action_space_dim)
        # critic's layer
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_layer(x), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.value_layer(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values



class ActorCritic:
    """Actor Critic Algorithm."""

    def __init__(self, obs_space_dim: int, action_space_dim: int, cuda: bool = False):
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu")

        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.lr = 3e-2
        self.eps = np.finfo(np.float32).eps.item()

        self.net = A2CPolicyNet(obs_space_dim, action_space_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
        
        pass 

    def sample_action(self, state: np.ndarray) -> float:
        state = torch.from_numpy(np.array([state])).float().to(self.device)
        probs, state_value = self.net(state)

        if self.cuda:
            probs = probs.cpu()
            state_value = state_value.cpu()
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # sample an action using the distribution
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), state_value))

        return action.item()

    def update(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()
        
        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # empty the state
        del self.rewards[:]
        del self.saved_actions[:]

    def save(self, wpath="weight.pt"):
        """Save the policy network weights"""
        torch.save(self.net.state_dict(), wpath)

    def load(self, wpath="weight.pt"):
        """Load the policy network weights"""
        self.net.load_state_dict(torch.load(wpath))