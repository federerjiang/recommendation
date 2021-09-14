from porise.envs.real import RealEnvBase 
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 
import pandas as pd

class ChainOfferEnv(RealEnvBase):
    """
    RealEnv need to load offline user features and calcualte online behavior features
    """
    def __init__(self, 
                rat_log_path, 
                user_vectors_map={},
                arm_vectors_map={}):
        super(ChainOfferEnv, self).__init__(
            rat_log_path=rat_log_path,
            user_vectors_map=user_vectors_map,
            arm_vectors_map=arm_vectors_map
        )
    
    @property
    def arm_feat_dim(self):
        return len(self.get_user_state()[1][0])

    @property
    def user_feat_dim(self):
        return len(self.get_user_state()[0])

    @property
    def n_arms(self):
        return self.action_space.n 

    def _load_log_data(self, rat_log_path):
        self.df = pd.read_csv(rat_log_path)
        # if 'landing_time' in self.df.columns:
        #     self.df.sort_values(by='landing_time', ascending=True, inplace=True)
        self.action_space = Discrete(len(self.df.service.unique()))
        self.max_steps = len(self.df.index) 
        self.max_users = self.df.easy_id.count()
        self.banner2id = self._banner2id()

    def _banner2id(self):
        banner2id = {}
        banners = self.df.service.unique()
        id_ = 0
        for banner in banners:
            banner2id[banner] = id_
            id_ += 1
        return banner2id
    
    def _arm_one_hot_mapping(self, r):
        arm_one_hot = np.zeros(self.action_space.n)
        arm_one_hot[int(r)] = 1.0
        return arm_one_hot

    def _get_user_id(self): 
        entry = self.df.iloc[self.steps_beyond_done]
        return entry.easy_id

    def get_user_state(self):
        # user state can be a tuple of (CDNA features, behavior features, and others)
        user_id = self._get_user_id()
        # step-1: get CDNA features
        next_user_cdna_vector = self.user_vectors_map[user_id]
        # step-2: get next state (user_CDNA_vector, arm)
        user_state = np.array(next_user_cdna_vector)
        arm_states = []
        for i_arm in range(self.action_space.n):
            arm_states.append(self._arm_one_hot_mapping(i_arm))
        return (user_state, arm_states)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """
        action: model selected banner id
        return:
         - state: features for next user
         - reward: reward of current user
         - done: wether all records have been run
         - info: 
        """
        # if isinstance(action, list):
        #     action = int(np.argmax(action))
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

        return self.state, reward, self.done, self.info

    def reset(self):
        super().reset()