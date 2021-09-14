from porise.envs.real import ChainOfferEnv
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 
import pandas as pd
import math 

class ChainOfferEnvV2(ChainOfferEnv):
    """
    RealEnv need to load offline user features and calcualte online behavior features
    """
    def __init__(self, 
                rat_log_path, 
                user_vectors_map={},
                arm_vectors_map={},
                T=1
        ):
        super(ChainOfferEnvV2, self).__init__(
            rat_log_path=rat_log_path,
            user_vectors_map=user_vectors_map,
            arm_vectors_map=arm_vectors_map
        )
        # initialze online user stats
        self.T = T 
        self.user_arm_neg_counts = dict()
        self.arm_prior_probs = np.ones(super().n_arms)

    @property
    def arm_feat_dim(self):
        return len(self.get_user_state()[0][1][0])

    @property
    def user_feat_dim(self):
        return len(self.get_user_state()[0][0])

    def _update_online_feats(self, user_id, arm_id, reward):
        if user_id not in self.user_arm_neg_counts:
            self.user_arm_neg_counts[user_id] = np.zeros(super().n_arms)
        if reward == 0:
            self.user_arm_neg_counts[user_id][arm_id] += 1
        else:
            self.user_arm_neg_counts[user_id][arm_id] = 0

    def get_arm_prior_probs(self):
        user_id = super()._get_user_id()
        if user_id not in self.user_arm_neg_counts:
            self.user_arm_neg_counts[user_id] = np.zeros(super().n_arms)
        arm_neg_counts = self.user_arm_neg_counts[user_id]
        # get function to caclaute prior probability based on consecutive negative counts on each arm
        prob_func = lambda x: 1 / (math.exp(x/self.T))
        self.arm_prior_probs = [prob_func(count) for count in arm_neg_counts]
        return self.arm_prior_probs

    def get_user_state(self):
        state = super().get_user_state()
        return (state, self.get_arm_prior_probs())

    def step(self, action):
        """
        action: model selected banner id
        return:
         - state: features for next user
         - reward: reward of current user
         - done: wether all records have been run
         - info: 
        """
        if isinstance(action, list):
            arm_prior_probs = self.get_arm_prior_probs()
            probs = []
            for arm_prior_prob, pred_prob in zip(arm_prior_probs, action):
                probs.append(arm_prior_prob * pred_prob)
            action = int(np.argmax(probs))
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward = 0
        entry = self.df.iloc[self.steps_beyond_done]
        item_id = self.banner2id[entry.service]
        if item_id != action:
            reward = 0
        else:
            reward = 1 if entry.reward > 0 else 0
            
        if self.steps_beyond_done == self.max_steps-1:
            self.done = True 
            self.steps_beyond_done = 0
        else:
            self.steps_beyond_done += 1
            self.state = self.get_user_state()

        # update user behavior and arm features
        self._update_online_feats(entry.easy_id, action, reward)
        return self.state, reward, self.done, self.info

    def reset(self):
        super().reset()
        self.user_arm_neg_counts = dict()
        self.arm_prior_probs = np.ones(super().n_arms)