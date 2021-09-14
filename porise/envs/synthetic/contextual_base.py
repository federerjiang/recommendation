from porise import Env 
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 
import itertools


class LinearEnv(Env):
    def __init__(self, n_arm, feature_dim, max_steps=1e5, noise_std=1.0):
        self.action_space = Discrete(n_arm)
        self.n_arm = n_arm
        self.feature_dim = feature_dim
        self.max_steps = max_steps
        self.h = self.get_reward_func()
        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        self.seed()
        # initialize arm features, rewards, and others.
        self.reset()
    
    def get_reward_func(self):
        a = np.random.randn(self.feature_dim)
        a /= np.linalg.norm(a, ord=2)
        return lambda x: 100*np.dot(a, x)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward = self.rewards[self.steps_beyond_done, action]
        best_action = self.best_actions_oracle[self.steps_beyond_done]
        regret = self.best_rewards_oracle[self.steps_beyond_done] - reward
        assert self.action_space.contains(best_action)
        assert regret >= 0
        self.info = {
            'best_arm_hit': best_action == action,
            'regret':  regret
        }
            
        if self.steps_beyond_done == self.max_steps-1:
            self.done = True 
            self.steps_beyond_done = 0
        else:
            self.steps_beyond_done += 1
            self.state = self.features[self.steps_beyond_done]

        return self.state, reward, self.done, self.info

    def reset(self):
        self.state = None
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}
        self.best_arm_hit = 0

        self.reset_features()
        self.reset_rewards(self.h)
    
    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.max_steps, self.n_arm, self.feature_dim)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.feature_dim).reshape(self.max_steps, self.n_arm, self.feature_dim)
        self.features = x

    def reset_rewards(self, h):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                h(self.features[t, k]) + self.noise_std*np.random.randn()\
                for t,k in itertools.product(range(self.max_steps), range(self.n_arm))
            ]
        ).reshape(self.max_steps, self.n_arm)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)