import numpy as np 
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist
from sklearn.linear_model import SGDClassifier, LogisticRegression
from scipy.optimize import minimize
from ..algo_base import AlgoBase


class OnlineLogisticRegression:
    """ Online LR
    """
    def __init__(self, lambda_, alpha, n_dim):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_dim = n_dim, 
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
        
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        
    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
    
    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)
    
    def fit(self, X, y):
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':True}).x
        self.m = self.w
        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)

    def predict_proba(self, X, mode='sample'):
        self.w = self.get_weights()
        if mode == 'sample':
            w = self.w 
        elif mode == 'expected':
            w = self.m 
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return proba


class LogisticTS(AlgoBase):
    """
    Logistic Thompson Sampling, each arm has its own parameters

    Parameters
    ----------
    n_arms: int
        Number of arms
    return_list: bool
        True: return the list of predicted logits of each arm
        False: return the index of the arm which has the largest logit

    References
    ----------
    
    """
    def __init__(self, 
                n_arms,
                lambda_=1, 
                alpha=5, 
                arm_feat_dim=0, 
                user_feat_dim=0, 
                return_list=True):
        super().__init__(n_arms)
        self.lambda_ = lambda_ 
        self.alpha = alpha 
        self.arm_feat_dim = arm_feat_dim
        self.user_feat_dim = user_feat_dim
        self.return_list = return_list
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = OnlineLogisticRegression(self.lambda_, self.alpha, self.arm_feat_dim+self.user_feat_dim)

    def predict(self, state):
        user_feat, arm_feat_list = state 
        pred_ts = [self.arms[i_arm].predict_proba(np.concatenate((user_feat, arm_feat_list[i_arm]), axis=0).reshape(1,-1))
            for i_arm in range(self.n_arms)]
        
        if self.return_list:
            return pred_ts
        else:
            return int(np.argmax(pred_ts))

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
            x = np.concatenate((state_list[i][0], state_list[i][1][action]), axis=0).reshape(1,-1)
            self.arms[action].fit(x, np.array([reward]))

    def reset(self):
        self.arms = dict()
        for i_arm in range(self.n_arms):
            self.arms[i_arm] = OnlineLogisticRegression(self.lambda_, self.alpha, self.arm_feat_dim+self.user_feat_dim)

    def default_prediction(self):
        return 