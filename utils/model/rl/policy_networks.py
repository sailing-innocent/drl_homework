import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal

class NormalMLPPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NormalMLPPolicyNet, self).__init__()

        self.input_layer = nn.Linear(state_dim, 512)
        self.hidden_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, action_dim)

        self.sigma = nn.Parameter(torch.zeros(action_dim))
        self.sigma.data.fill_(math.log(1.0))

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        mu = self.output_layer(x)
        return Normal(loc=mu, scale=self.sigma)


class SharedNormalMLPPolicyNet(nn.Module):
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        super().__init__()
        hidden_space1 = 16
        hidden_space2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dim, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh()
        )
        # Policy Mean Specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dim)
        )

        # Policy Stardard Derivitive Liear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned On the observation, returns the mean and standard derivative
        of a normal distribution from which an action is sampled

        Args:
            x: Observation from the environment
        
        Returns:
            action_mean: predicted mean of the normal distribution
            action_stddevs: predict the standard derivation of the normal distribution
        """

        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs