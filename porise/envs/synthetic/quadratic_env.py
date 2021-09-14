from porise.envs.synthetic.linear_env import LinearEnv

import numpy as np


class QuadraticEnv(LinearEnv):
    
    def __init__(self, n_arm, feature_dim, max_steps=1e5, noise_std=1.0):
        super(QuadraticEnv, self).__init__(n_arm, feature_dim, max_steps, noise_std)
        self.h = self.get_reward_func()
        self.reset_rewards(self.h)

    def get_reward_func(self):
        a = np.random.randn(self.feature_dim)
        a /= np.linalg.norm(a, ord=2)
        return lambda x: 100*np.dot(a, x)**2