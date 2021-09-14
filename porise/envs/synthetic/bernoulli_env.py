from porise import Env 
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 

class BernoulliEnv(Env):
    def __init__(self, n_arm, arm_probs, max_steps=1e5):
        self.arm_probs = arm_probs
        assert np.sum(self.arm_probs) == 1
        self.action_space = Discrete(n_arm)
        self.max_steps = max_steps

        self.seed()
        self.state = None
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward = 0
        if not self.done:
            reward = np.random.binomial(1, self.arm_probs[action])
            self.steps_beyond_done += 1
        if self.steps_beyond_done == self.max_steps:
            self.done = True 
            self.reset()

        return self.state, reward, self.done, self.info

    def reset(self):
        self.state = None
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}