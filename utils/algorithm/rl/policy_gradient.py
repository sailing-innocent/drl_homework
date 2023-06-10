from utils.model.rl.policy_networks import SharedNormalMLPPolicyNet
import numpy as np
import torch
from torch.distributions.normal import Normal

from utils.optim.ngd import NGD

class REINFORCE:
    """REINFORCE (Vanilla Policy Gradient) algorithm"""

    def __init__(self, obs_space_dim: int, action_space_dim: int, cuda: bool = False):
        """Initialize the agent that learns a policy using REINFORCE algorithm
        """
        
        # Hyperparameters
        self.lr = 5e-4 # learning rate
        self.gamma = 0.99 # the reward decay
        self.eps = 1e-6
        self.cuda = cuda

        self.probs = []
        self.rewards = []
        self.net = SharedNormalMLPPolicyNet(obs_space_dim, action_space_dim)
        if cuda:
            self.net = self.net.cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.97)
        # pron to NaN error
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, momentum=0.97)


    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation

        Args:
            state: Observation from the environment
        
        Returns:
            action: Actions to be performed
        """

        state = torch.Tensor(np.array([state]))
        if self.cuda:
            state = state.cuda()
        
        action_means, action_stddevs = self.net(state)

        if self.cuda:
            action_means = action_means.cpu()
            action_stddevs = action_stddevs.cpu()

        # create a normal distribution from predicted mean and standard derivation

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob) # used for loss 

        return action

    def update(self):
        """Update the Polic Network weights"""
        running_g = 0
        gs = []

        # Discounted return (backwards) [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs) # change gs to tensor

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1) # J(\theta), now loss is tensor, too

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward() # tensor.backward() will calculate the gradient
        self.optimizer.step()

        # Empty the state
        self.probs = []
        self.rewards = []

    def save(self, wpath="weight.pt"):
        """Save the policy network weights"""
        torch.save(self.net.state_dict(), wpath)

    def load(self, wpath="weight.pt"):
        """Load the policy network weights"""
        self.net.load_state_dict(torch.load(wpath))




class NPG:
    """Natural Policy Gradient algorithm"""

    def __init__(self, obs_space_dim: int, action_space_dim: int, cuda: bool = False):
        """Initialize the agent that learns a policy using REINFORCE algorithm
        """
        
        # Hyperparameters
        self.lr = 5e-4 # learning rate
        self.gamma = 0.99
        self.eps = 1e-6
        self.cuda = cuda

        self.probs = []
        self.rewards = []
        self.net = SharedNormalMLPPolicyNet(obs_space_dim, action_space_dim)
        if cuda:
            self.net = self.net.cuda()
        # self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        self.optimizer = NGD(self.net.parameters(), lr=self.lr)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation

        Args:
            state: Observation from the environment
        
        Returns:
            action: Actions to be performed
        """

        state = torch.Tensor(np.array([state]))
        if self.cuda:
            state = state.cuda()
        
        action_means, action_stddevs = self.net(state)

        torch.where(torch.isnan(action_means), torch.full_like(action_means, 0), 
            action_means)
        torch.where(torch.isnan(action_stddevs), torch.full_like(action_stddevs, 0), 
            action_stddevs)
        if self.cuda:
            action_means = action_means.cpu()
            action_stddevs = action_stddevs.cpu()

        # create a normal distribution from predicted mean and standard derivation

        distrib = Normal(loc = action_means[0] + self.eps, 
            scale=action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Update the Polic Network weights"""
        running_g = 0
        gs = []

        # Discounted return (backwards) [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty the state
        self.probs = []
        self.rewards = []

    def save(self, wpath="weight.pt"):
        """Save the policy network weights"""
        torch.save(self.net.state_dict(), wpath)

    def load(self, wpath="weight.pt"):
        """Load the policy network weights"""
        self.net.load_state_dict(torch.load(wpath))