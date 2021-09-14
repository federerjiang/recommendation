import numpy as np 
from collections import defaultdict
from ..algo_base import AlgoBase

class BetaThompsonSampling(AlgoBase):
    """
    Thompson Sampling with beta distribution

    Parameters
    ----------
    n_arms: int
        Number of arms
    return_list: bool
        True: return the list of predicted logits of each arm
        False: return the index of the arm which has the largest logit

    References
    ----------
    Chapelle, Olivier, and Lihong Li. 
    Algoithm 2 in "An empirical evaluation of thompson sampling." Advances in neural information processing systems 24 (2011)
    
    """
    def __init__(self, n_arms, return_list=True):
        super().__init__(n_arms)
        self.return_list = return_list
        self.weights = defaultdict(dict)
        for i_arm in range(self.n_arms):
            self.weights[i_arm]["alpha"] = 1
            self.weights[i_arm]["beta"] = 1

    def predict(self, state):
        pred_logits = [np.random.beta(self.weights[i_arm]["alpha"], self.weights[i_arm]["beta"]) 
            for i_arm in range(self.n_arms)]
        
        if self.return_list:
            return pred_logits
        else:
            return int(np.argmax(pred_logits))

    def train(self, action_list, state_list, reward_list):
        """
        Batch update alpha and beta values

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
            self.weights[action]["alpha"] += reward
            self.weights[action]["beta"] += 1 - reward

    def reset(self):
        for i_arm in range(self.n_arms):
            self.weights[i_arm]["alpha"] = 1
            self.weights[i_arm]["beta"] = 1
    
    def default_prediction(self):
        return 