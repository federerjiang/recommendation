from porise import Env 
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 
import itertools


class LinearEnv(Env):
    def __init__(self, 
                n_arms, 
                user_feat_dim=100, 
                max_steps=int(1e5), 
                noise_std=5.0):
        self.action_space = Discrete(n_arms)
        self.n_arms = n_arms
        self.user_feat_dim = user_feat_dim
        self.arm_feat_dim = n_arms
        self.max_steps = max_steps
        self.h = self.get_reward_func()
        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        self.seed()
        # initialize arm features, rewards, and others.
        self.reset()

    def get_reward_func(self):
        a = np.random.randn(self.user_feat_dim+self.arm_feat_dim)
        a /= np.linalg.norm(a, ord=2)
        return lambda x: 100*np.dot(a, x)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_user_state(self):
        return ([], self.features[self.steps_beyond_done])

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
            self.state = self.get_user_state()
        if action == best_action:
            reward = 1
        else:
            reward = 0
        return self.state, reward, self.done, self.info

    def reset(self):
        self.state = None
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}
        self.best_arm_hit = 0

        self.reset_features()
        self.reset_rewards(self.h)
    
    def _arm_one_hot_mapping(self, r):
        arm_one_hot = np.zeros(self.action_space.n)
        arm_one_hot[int(r)] = 1.0
        return arm_one_hot

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        user_feats = np.random.randn(self.max_steps, self.user_feat_dim)
        features = []
        for uid in range(self.max_steps):
            user_feat = user_feats[uid]
            user_feat /= np.linalg.norm(user_feat, ord=2)
            arm_feats = []
            for i_arm in range(self.n_arms):
                arm_feats.append(np.concatenate((user_feat, self._arm_one_hot_mapping(i_arm)), axis=0))
            features.append(arm_feats)
        self.features = np.array(features).reshape(self.max_steps, self.n_arms, self.user_feat_dim+self.arm_feat_dim)

    def reset_rewards(self, h):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                h(self.features[t, k]) + self.noise_std*np.random.randn()\
                for t,k in itertools.product(range(self.max_steps), range(self.n_arms))
            ]
        ).reshape(self.max_steps, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)