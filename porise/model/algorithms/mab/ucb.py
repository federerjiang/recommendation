import numpy as np
from collections import defaultdict
from ..algo_base import AlgoBase

class UCB1(AlgoBase):
    """
    UCB1 
    for arm i at time t,
    UCB_i^(t) = hat_mu_i^(t) +  sqrt( [ alpha * log (t) ] /  N_i(t)  )
    N_i(t) : up to time t, the # of times arm i was picked 
    select argmax (UBC_i(t)) over K arms, observe reward x_i(t)

    Parameters
    ----------
    n_arms: int
        Number of arms
    alpha: float
        Hyper parameter for adjusting the exploration
    return_list: bool
        True: return the list of predicted logits of each arm
        False: return the index of the arm which has the largest logit
    """
    def __init__(self, n_arms, alpha=2, return_list=True):
        super().__init__(n_arms)
        self.return_list = return_list
        self.weights = defaultdict(dict)
        for i_arm in range(self.n_arms):
            self.weights[i_arm]["mu"] = 0
            self.weights[i_arm]["N"] = 1
        self.alpha = alpha 
        self.t = 0
        
    def predict(self, state):
        pred_ucb = [self.weights[i_arm]["mu"] +\
            np.sqrt(self.alpha * np.log(self.t) / self.weights[i_arm]["N"]) 
            for i_arm in range(self.n_arms)]
        
        if self.return_list:
            return pred_ucb
        else:
            return int(np.argmax(pred_ucb))

    def train(self, action_list, state_list, reward_list):
        """
        Batch update UCB1 model weights

        Parameters
        ----------
        action_list: [1, 2, 4] list 
            The arms which are seleted by the system
        reward_list: [1, 0, 1] list
            The corresponding rewards of the selected arms
        """
        assert len(action_list) == len(reward_list)

        for i in range(len(action_list)):
            action = action_list[i]
            reward = reward_list[i]
            n = self.weights[action]["N"]
            self.weights[action]["N"] += 1
            self.weights[action]["mu"] = (n*self.weights[action]["mu"]+reward) / (n+1)
            self.t += 1

    def reset(self):
        for i_arm in range(self.n_arms):
            self.weights[i_arm]["mu"] = 0
            self.weights[i_arm]["N"] = 1
        self.t = 0

    def default_prediction(self):
        return 