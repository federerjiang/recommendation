from porise import Env 
from porise.envs.utils import seeding
from porise.envs.utils.discrete import Discrete

import numpy as np 
import pandas as pd

class RealEnvBase(Env):
    """
    RealEnvBase need to load offline user features and calcualte online behavior features
    """
    def __init__(self, 
                rat_log_path, 
                user_vectors_map={},
                arm_vectors_map={}):
        self._load_log_data(rat_log_path=rat_log_path)
        self.user_vectors_map = user_vectors_map
        self.arm_vectors_map = arm_vectors_map

        self.seed()
        self.state = None
        self.user_online_state = None
        self.arm_online_state = None 
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}
    
    @property
    def arm_feat_dim(self):
        raise NotImplementedError

    @property
    def user_feat_dim(self):
        raise NotImplementedError

    @property
    def n_arms(self):
        raise NotImplementedError

    def _load_log_data(self, rat_log_path):
        raise NotImplementedError

    def _update_online_feats(self):
        pass 

    def get_user_state(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]        

    def reset(self):
        self.state = None
        self.user_online_state = None
        self.arm_online_state = None 
        self.steps_beyond_done = 0
        self.done = False 
        self.info = {}