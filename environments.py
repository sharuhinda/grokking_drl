from collections import namedtuple
import numpy as np



Transition = namedtuple('StepResult', ['proba', 'next_state', 'reward', 'is_terminal'])

class Environment:
    
    P = {
        0: {
            0: [Transition(1.0, 0, 0., True)]
        }
    }
    
    def __init__(self, start_state=0) -> None:
        self.start_state = start_state
        self.cur_state = start_state

    def reset(self):
        self.cur_state = self.start_state
    
    def get_transitions(self, s=0, a=0):
        return self.P[s][a]

    def step(self, s=0, a=0):
        return self.P[s][a]
    
    def step_with_choice(self, s=0, a=0):
        result = self.step(s, a)
        p = np.array([x.proba for x in result])
        i = np.random.choice(len(result), p=p)
        return result[i]
    

class BW(Environment):
    """
    Bandit walk environment
    """
    
    # outer keys are states
    # inner keys are actions (0=left, 1=right)
    # values for each action is a list of tuples for all possible transitions for this state-action pair
    # tuples consist of: probability of transition, next state id, reward value, flag indicating whether the next state is terminal or not
    P = {
        0: { # terminal state is linked to itself
            0: [Transition(1.0, 0, 0., True)],
            1: [Transition(1.0, 0, 0., True)]
        },
        1: {
            0: [Transition(1.0, 0, 0., True)],
            1: [Transition(1.0, 2, 1., True)]
        },
        2: { # terminal state is linked to itself
            0: [Transition(1.0, 2, 0., True)],
            1: [Transition(1.0, 2, 0., True)]
        },
    }

    def __init__(self, start_state=1) -> None:
        super().__init__(start_state)


class BSW(Environment):
    """
    Bandit Slippery Walk environment
    """

    P = {
        0: { # terminal state is linked to itself
            0: [Transition(1.0, 0, 0., True)],
            1: [Transition(1.0, 0, 0., True)]
        },
        1: {
            0: [Transition(0.8, 0, 0., True), Transition(0.2, 2, 1., True)],
            1: [Transition(0.8, 2, 1., True), Transition(0.2, 0, 0., True)]
        },
        2: { # terminal state is linked to itself
            0: [Transition(1.0, 2, 0., True)],
            1: [Transition(1.0, 2, 0., True)]
        },
    }

    def __init__(self, start_state=1) -> None:
        super().__init__(start_state)


class FL(Environment):
    """
    Frosen Lake environment
    Actions: 0=left, 1=down, 2=right, 3=up
    """

    P = {
        0: { # starting state
            0: [Transition(2/3, 0, 0., False), Transition(1/3, 4, 0., False)],
            1: [Transition(1/3, 4, 0., False), Transition(1/3, 0, 0., False), Transition(1/3, 1, 0., False)],
            2: [Transition(1/3, 1, 0., False), Transition(1/3, 0, 0., False), Transition(1/3, 4, 0., False)],
            3: [Transition(2/3, 0, 0., False), Transition(1/3, 1, 0., False)]
        },
        1: {
            0: [Transition(1/3, 0, 0., False), Transition(1/3, 1, 0., False), Transition(1/3, 5, 0., True)],
            1: [Transition(1/3, 5, 0., True), Transition(1/3, 0, 0., False), Transition(1/3, 2, 0., False)],
            2: [Transition(1/3, 2, 0., False), Transition(1/3, 1, 0., False), Transition(1/3, 5, 0., True)],
            3: [Transition(1/3, 1, 0., False), Transition(1/3, 0, 0., False), Transition(1/3, 2, 0., False)]
        },
        2: {
            0: [Transition(1/3, 1, 0., False), Transition(1/3, 2, 0., False), Transition(1/3, 6, 0., False)],
            1: [Transition(1/3, 6, 0., False), Transition(1/3, 1, 0., False), Transition(1/3, 3, 0., False)],
            2: [Transition(1/3, 3, 0., False), Transition(1/3, 2, 0., False), Transition(1/3, 6, 0., False)],
            3: [Transition(1/3, 2, 0., False), Transition(1/3, 1, 0., False), Transition(1/3, 3, 0., False)]
        },
        3: {
            0: [Transition(1/3, 2, 0., False), Transition(1/3, 3, 0., False), Transition(1/3, 7, 0., True)],
            1: [Transition(1/3, 7, 0., True), Transition(1/3, 2, 0., False), Transition(1/3, 3, 0., False)],
            2: [Transition(2/3, 3, 0., False), Transition(1/3, 7, 0., True)],
            3: [Transition(2/3, 3, 0., False), Transition(1/3, 2, 0., False)]
        },
        4: {
            0: [Transition(1/3, 4, 0., False), Transition(1/3, 0, 0., False), Transition(1/3, 8, 0., False)],
            1: [Transition(1/3, 8, 0., False), Transition(1/3, 4, 0., False), Transition(1/3, 5, 0., True)],
            2: [Transition(1/3, 5, 0., True), Transition(1/3, 0, 0., False), Transition(1/3, 8, 0., False)],
            3: [Transition(1/3, 0, 0., False), Transition(1/3, 4, 0., False), Transition(1/3, 5, 0., True)]
        },
        5: { # terminal state (hole)
            0: [Transition(1.0, 5, 0., True)],
            1: [Transition(1.0, 5, 0., True)],
            2: [Transition(1.0, 5, 0., True)],
            3: [Transition(1.0, 5, 0., True)]
        },
        6: {
            0: [Transition(1/3, 5, 0., True), Transition(1/3, 2, 0., False), Transition(1/3, 10, 0., False)],
            1: [Transition(1/3, 10, 0., False), Transition(1/3, 5, 0., True), Transition(1/3, 7, 0., True)],
            2: [Transition(1/3, 7, 0., True), Transition(1/3, 2, 0., False), Transition(1/3, 10, 0., False)],
            3: [Transition(1/3, 2, 0., False), Transition(1/3, 5, 0., True), Transition(1/3, 7, 0., True)]
        },
        7: { # terminal state (hole)
            0: [Transition(1.0, 7, 0., True)],
            1: [Transition(1.0, 7, 0., True)],
            2: [Transition(1.0, 7, 0., True)],
            3: [Transition(1.0, 7, 0., True)]
        },
        8: {
            0: [Transition(1/3, 8, 0., False), Transition(1/3, 4, 0., False), Transition(1/3, 12, 0., True)],
            1: [Transition(1/3, 12, 0., True), Transition(1/3, 8, 0., False), Transition(1/3, 9, 0., False)],
            2: [Transition(1/3, 9, 0., False), Transition(1/3, 4, 0., False), Transition(1/3, 12, 0., True)],
            3: [Transition(1/3, 4, 0., False), Transition(1/3, 8, 0., False), Transition(1/3, 9, 0., False)]
        },
        9: {
            0: [Transition(1/3, 8, 0., False), Transition(1/3, 5, 0., True), Transition(1/3, 13, 0., False)],
            1: [Transition(1/3, 13, 0., False), Transition(1/3, 8, 0., False), Transition(1/3, 10, 0., False)],
            2: [Transition(1/3, 10, 0., False), Transition(1/3, 5, 0., True), Transition(1/3, 13, 0., False)],
            3: [Transition(1/3, 5, 0., True), Transition(1/3, 8, 0., False), Transition(1/3, 10, 0., False)]
        },
        10: {
            0: [Transition(1/3, 9, 0., False), Transition(1/3, 6, 0., False), Transition(1/3, 14, 0., False)],
            1: [Transition(1/3, 14, 0., False), Transition(1/3, 9, 0., False), Transition(1/3, 11, 0., True)],
            2: [Transition(1/3, 11, 0., True), Transition(1/3, 6, 0., False), Transition(1/3, 14, 0., False)],
            3: [Transition(1/3, 6, 0., False), Transition(1/3, 9, 0., False), Transition(1/3, 11, 0., True)]
        },
        11: { # terminal state (hole)
            0: [Transition(1.0, 11, 0., True)],
            1: [Transition(1.0, 11, 0., True)],
            2: [Transition(1.0, 11, 0., True)],
            3: [Transition(1.0, 11, 0., True)]
        },
        12: { # terminal state (hole)
            0: [Transition(1.0, 12, 0., True)],
            1: [Transition(1.0, 12, 0., True)],
            2: [Transition(1.0, 12, 0., True)],
            3: [Transition(1.0, 12, 0., True)]
        },
        13: {
            0: [Transition(1/3, 12, 0., True), Transition(1/3, 9, 0., False), Transition(1/3, 13, 0., False)],
            1: [Transition(1/3, 13, 0., False), Transition(1/3, 12, 0., True), Transition(1/3, 14, 0., False)],
            2: [Transition(1/3, 14, 0., False), Transition(1/3, 9, 0., False), Transition(1/3, 13, 0., False)],
            3: [Transition(1/3, 9, 0., False), Transition(1/3, 12, 0., True), Transition(1/3, 14, 0., False)]
        },
        14: {
            0: [Transition(1/3, 13, 0., False), Transition(1/3, 10, 0., False), Transition(1/3, 14, 0., False)],
            1: [Transition(1/3, 14, 0., False), Transition(1/3, 13, 0., False), Transition(1/3, 15, 1., True)],
            2: [Transition(1/3, 15, 1., True), Transition(1/3, 10, 0., False), Transition(1/3, 14, 0., False)],
            3: [Transition(1/3, 10, 0., False), Transition(1/3, 13, 0., False), Transition(1/3, 15, 1., True)]
        },
        15: { # terminal state (goal)
            0: [Transition(1.0, 15, 0., True)],
            1: [Transition(1.0, 15, 0., True)],
            2: [Transition(1.0, 15, 0., True)],
            3: [Transition(1.0, 15, 0., True)]
        },
    }

    def __init__(self, start_state=0) -> None:
        super().__init__(start_state)


class SWF(Environment):
    """
    Slippery Walk Five environment
    Actions: 0=left, 1=right
    """

    P = {
        0: { # terminal state (hole)
            0: [Transition(1.0, 0, 0., True)],
            1: [Transition(1.0, 0, 0., True)]
        },
        1: {
            0: [Transition(0.5, 0, 0., True), Transition(0.33, 1, 0., False), Transition(0.17, 2, 0., False)],
            1: [Transition(0.5, 2, 0., False), Transition(0.33, 1, 0., False), Transition(0.17, 0, 0., True)]
        },
        2: {
            0: [Transition(0.5, 1, 0., False), Transition(0.33, 2, 0., False), Transition(0.17, 3, 0., False)],
            1: [Transition(0.5, 3, 0., False), Transition(0.33, 2, 0., False), Transition(0.17, 1, 0., False)]
        },
        3: {
            0: [Transition(0.5, 2, 0., False), Transition(0.33, 3, 0., False), Transition(0.17, 4, 0., False)],
            1: [Transition(0.5, 4, 0., False), Transition(0.33, 3, 0., False), Transition(0.17, 2, 0., False)]
        },
        4: {
            0: [Transition(0.5, 3, 0., False), Transition(0.33, 4, 0., False), Transition(0.17, 5, 0., False)],
            1: [Transition(0.5, 5, 0., False), Transition(0.33, 4, 0., False), Transition(0.17, 3, 0., False)]
        },
        5: {
            0: [Transition(0.5, 4, 0., False), Transition(0.33, 5, 0., False), Transition(0.17, 6, 1., True)],
            1: [Transition(0.5, 6, 1., True), Transition(0.33, 5, 0., False), Transition(0.17, 4, 0., False)]
        },
        6: { # terminal state (goal)
            0: [Transition(1.0, 6, 0., True)],
            1: [Transition(1.0, 6, 0., True)]
        },
    }

    def __init__(self, start_state=3) -> None:
        super().__init__(start_state)

