import numpy as np


class Policy:
    def __init__(self, P=None, d=None) -> None:
        if d is None:
            #num_actions = 
            self.pi = {}



def policy_evaluation(pi, P, gamma=0.9, epsilon=1e-5):
    """
    Arguments:
        pi - policy => function to map states to actions
        P - dict of states, actions and transitions' parameters
    """
    cycles = 0
    prev_v = np.zeros(len(P))
    while True:
        cycles += 1
        v = np.zeros(len(P))
        for s in P.keys():
            # this cycle goes through all transitions for action returned by policy pi for state s
            for proba, next_s, reward, is_terminal in P[s][pi[s]]:
                v[s] += proba * (reward + gamma * prev_v[next_s] * (not is_terminal))

        if np.max(np.abs(prev_v - v)) < epsilon:
            break
        prev_v = v.copy()

    return v, cycles




# =======================================================
if __name__ == '__main__':
    pass