import numpy as np 
from scipy import linalg
from ..algo_base import AlgoBase


class HybridArm:
    #Representation of each arm. Has a set of arm-specific features of size d, common features k
    def __init__(self, id, d, k, alpha):
        self.id = id
        self.d = d
        self.k = k
        self.alpha = alpha
        #Li lines 8-10
        self.A = np.identity(self.d)
        self.Ainv = linalg.inv(self.A)
        self.B = np.zeros((self.d, self.k))
        self.b = np.zeros((self.d, 1))

    def getAinv(self):
        return self.Ainv
    def getB(self):
        return self.B
    def getb(self):
        return self.b
    def getID(self):
        return self.id
    
    def getP(self, A0inv, b0, betaHat, z, x):
        #Li Lines 12-14, args are numpy arrays
        self.thetaHat = np.dot(self.Ainv, self.b - np.dot(self.B, betaHat))
        
        self.s1 = np.dot(np.transpose(z), np.dot(A0inv, z))
        self.s2 = np.dot(np.transpose(z), np.dot(A0inv, np.dot(np.transpose(self.B), np.dot(self.Ainv, x))))
        self.s3 = np.dot(np.transpose(x), np.dot(self.Ainv, x))
        self.s4 = np.dot(np.transpose(x), np.dot(self.Ainv, np.dot(self.B, np.dot(A0inv, np.dot(np.transpose(self.B), np.dot(self.Ainv, x))))))

        self.s = self.s1 - 2*self.s2 + self.s3 + self.s4 #Li line 13

        self.p = np.dot(np.transpose(z), betaHat) + np.dot(np.transpose(x), self.thetaHat) + self.alpha*np.sqrt(self.s)
        return self.p

    def update(self, z, x, reward):
        self.A += np.dot(x, np.transpose(x))
        self.Ainv = linalg.inv(self.A)
        self.B += np.dot(x, np.transpose(z))
        self.b += reward * x


class HybridLinUCB(AlgoBase):
    """
    Hybrid Linear UCB

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
    def __init__(self, n_arms, alpha, arm_feat_dim=0, user_feat_dim=0, return_list=True):
        super().__init__(n_arms)
        self.alpha = alpha 
        self.arm_feat_dim = arm_feat_dim
        self.user_feat_dim = user_feat_dim
        self.return_list = return_list        
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = HybridArm(i_arm, self.arm_feat_dim, self.user_feat_dim, self.alpha)
        
        self.A0 = np.identity(self.user_feat_dim)
        self.A0inv = linalg.inv(self.A0)
        self.b0 = np.zeros((self.user_feat_dim, 1))
        self.betaHat = np.dot(self.A0inv, self.b0)

    def predict(self, state):
        user_feat, arm_feat_list = state
        pred_ucb = [self.arms[i_arm].getP(self.A0inv, self.b0, self.betaHat, user_feat, arm_feat_list[i_arm])
            for i_arm in range(self.n_arms)]
        # user_cdna, user_stat, arm_feat_list = state 
        # user_feat = np.concatenate((user_cdna, user_stat), axis=0)
        # pred_ucb = [self.arms[i_arm].getP(self.A0inv, self.b0, self.betaHat, user_feat, arm_feat_list[i_arm])
        #     for i_arm in range(self.n_arms)]
        
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
            x = state_list[i][1][action].reshape(-1, 1)
            z = state_list[i][0].reshape(-1, 1)
            # x = state_list[i][2][action].reshape(-1, 1)
            # user_cdna, user_stat = state_list[i][0], state_list[i][1]
            # z = np.concatenate((user_cdna, user_stat), axis=0).reshape(-1, 1)

            
            current_arm = self.arms[action]
            #lines 17-18
            self.A0 += np.dot(np.transpose(current_arm.getB()),  np.dot(current_arm.getAinv(), current_arm.getB()))
            self.b0 += np.dot(np.transpose(current_arm.getB()), np.dot(current_arm.getAinv(), current_arm.getb()))
            #Update the arm-specific matrices: lines 19-21
            self.arms[action].update(z, x, reward)
            #Update the general matrices again: lines 22-23
            current_arm = self.arms[action]
            self.A0 += np.dot(z, np.transpose(z))
            self.A0 -= np.dot(np.transpose(current_arm.getB()), np.dot(current_arm.getAinv(), current_arm.getB()))
            self.b0 += reward * z
            self.b0 -= np.dot(np.transpose(current_arm.getB()), np.dot(current_arm.getAinv(), current_arm.getb()))
        self.A0inv = linalg.inv(self.A0)
        self.betaHat = np.dot(self.A0inv, self.b0)

    def reset(self):
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = HybridArm(i_arm, self.arm_feat_dim, self.user_feat_dim, self.alpha)
        
        self.A0 = np.identity(self.user_feat_dim)
        self.A0inv = linalg.inv(self.A0)
        self.b0 = np.zeros((self.user_feat_dim, 1))
        self.betaHat = np.dot(self.A0inv, self.b0)

    def default_prediction(self):
        return 