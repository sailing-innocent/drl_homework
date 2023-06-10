# The epsilon greedy algorithm 

import numpy as np 

def eps_greedy(Qa, eps=0.1):
    # choose an action
    if np.random.random() < eps:
        A = np.random.randint(10)
    else:
        A = np.argmax(Qa)

    return A