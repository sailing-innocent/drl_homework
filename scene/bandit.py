# The k-armed bandit

import numpy as np 

class Bandit:
    def __init__(self, k):
        self.k = k
        # one shot case
        self.q_star_means = np.zeros(k)
        self.q_star_means[k//2] = 1
        self.q_star_stds = np.ones(k)

    def step(self, action):
        return np.random.normal(self.q_star_means[action], self.q_star_stds[action])


class DynamicBandit:
    def __init__(self, k):
        self.k = k
        # one shot case
        self.q_star_means = np.zeros(k)
        self.q_star_means[k//2] = 1
        self.q_star_stds = np.ones(k)

    def step(self, action, t):
        # after some time, the q_star_means changes
        if t % 100:
            self.q_star_means = np.zeros(self.k)
            self.q_star_means[self.k//3] = 1
    
        return np.random.normal(self.q_star_means[action], self.q_star_stds[action])
