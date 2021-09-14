import numpy as np 

from porise.model.algorithms.cmab import HybridLinUCB

class BayesianHybridLinUCB(HybridLinUCB):
    
    def __init__(self, n_arms, alpha, arm_feat_dim=0, user_feat_dim=0, return_list=True):
        super(BayesianHybridLinUCB, self).__init__(
                n_arms=n_arms,
                alpha=alpha,
                arm_feat_dim=arm_feat_dim,
                user_feat_dim=user_feat_dim,
                return_list=return_list)
    
    def predict(self, state):
        (user_feat, arm_feat_list), arm_prior_probs = state
        preds = [self.arms[i_arm].getP(self.A0inv, self.b0, self.betaHat, user_feat, arm_feat_list[i_arm])
            for i_arm in range(self.n_arms)]
        probs = []
        for arm_prior_prob, pred_prob in zip(arm_prior_probs, preds):
                probs.append(arm_prior_prob * pred_prob)
        if self.return_list:
            return probs 
        else:
            return int(np.argmax(probs))

    def train(self, action_list, state_list, reward_list):
        super().train(
            action_list=action_list,
            state_list=[state[0] for state in state_list],
            reward_list=reward_list
        )