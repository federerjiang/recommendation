import numpy as np 
from scipy import linalg
from ..algo_base import AlgoBase


class LinArm:
    #Representation of each arm. Has a set of arm-specific features of size d
    def __init__(self, id, d, alpha):
        self.id = id
        self.d = d
        self.alpha = alpha
        #Li Alg1 lines 5-6
        self.A = np.identity(self.d)
        self.Ainv = linalg.inv(self.A)
        self.b = np.zeros((self.d, 1))

    def getA(self):
        return self.A

    def getb(self):
        return self.b

    def getID(self):
        return self.id
    
    def getP(self, x):
        #Li Alg1 Lines 8-9, args are numpy arrays
        # x is shape of (d, 1)
        self.thetaHat = np.dot(self.Ainv, self.b)
        s = np.dot(np.transpose(x), np.dot(self.Ainv, x))
        self.p = np.dot(np.transpose(self.thetaHat), x) + self.alpha*np.sqrt(s)
        return self.p

    def update(self, x, reward):
        #Li Alg1 Line 12
        self.A += np.dot(x, np.transpose(x))
        # pre-calculate Ainv for Alg1 Line 8-9
        self.Ainv = linalg.inv(self.A)
        self.b += reward * x


class LinUCB(AlgoBase):
    """
    Linear UCB

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
    def __init__(self, n_arms, alpha=2, arm_feat_dim=0, user_feat_dim=0, return_list=True):
        super().__init__(n_arms)
        self.alpha = alpha 
        self.arm_feat_dim = arm_feat_dim
        self.user_feat_dim = user_feat_dim
        self.return_list = return_list
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = LinArm(i_arm, self.arm_feat_dim+self.user_feat_dim, self.alpha)

    def predict(self, state):
        user_feat, arm_feat_list = state 
        pred_ucb = [self.arms[i_arm].getP(np.concatenate((user_feat, arm_feat_list[i_arm]), axis=0))
            for i_arm in range(self.n_arms)]
        
        if self.return_list:
            return pred_ucb
        else:
            return int(np.argmax(pred_ucb))

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
        assert len(action_list) == len(state_list)
        for i in range(len(action_list)):
            action = action_list[i]
            reward = reward_list[i]
            x = np.concatenate((state_list[i][0], state_list[i][1][action]), axis=0).reshape(-1, 1)
            self.arms[action].update(x, reward)

    def reset(self):
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = LinArm(i_arm, self.arm_feat_dim+self.user_feat_dim, self.alpha)

    def default_prediction(self):
        return 