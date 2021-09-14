import numpy as np 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from scipy import linalg
from ..algo_base import AlgoBase
from porise.model.replay_memory import PER, Transition
from porise.model.dnn import LogisticRegression
from porise.model.util import FTRL


class FTRL_LR(AlgoBase):
    """
    FTRL LR

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
    def __init__(self, 
                n_arms, 
                arm_feat_dim=0, 
                user_feat_dim=0, 
                return_list=True,
                memory_size=int(1e5),
                use_cuda=False,
                prio_a=0.6,
                prio_beta=0.4,
                prio_e=0.001,
                beta_increment_per_sampling=0.4e-6,
                batch_size=128,
                epochs=10
                ):
        super().__init__(n_arms)
        self.arm_feat_dim = arm_feat_dim
        self.user_feat_dim = user_feat_dim
        self.return_list = return_list
        
        self.memory_size = memory_size
        self.a = prio_a
        self.beta = prio_beta
        self.beta_increment_per_sampling = beta_increment_per_sampling 
        self.per_memory = PER(
            memory_size=self.memory_size,
            a=self.a,
            beta=self.beta,
            beta_increment_per_sampling=self.beta_increment_per_sampling
        )

        self.batch_size = batch_size
        self.epochs = epochs

        self.use_cuda = use_cuda 
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        self.model = LogisticRegression(input_size=self.arm_feat_dim+self.user_feat_dim,
                                        output_size=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = FTRL(self.model.parameters(),
                            alpha=1.0,
                            beta=1.0,
                            l1=1.0,
                            l2=1.0)

    def predict(self, state):
        user_feat, arm_feat_list = state
        pred = []
        with torch.no_grad():
            for i_arm in range(self.n_arms):
                x = np.concatenate((user_feat, arm_feat_list[i_arm]), axis=0)
                x = torch.FloatTensor(x).reshape(1,-1).to(self.device)
                y = self.model(x)
                pred.append(y)
            pred_t = torch.cat(pred)
            probs = F.softmax(pred_t, dim=0).reshape(-1)
        # print(probs)
        m = torch.distributions.categorical.Categorical(probs=probs)
        sampled_action = m.sample().item()
        probs[sampled_action] = 1.0
        if self.return_list:
            return probs 
        else:
            return sampled_action

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
        # collect the experiences, train at least once, and store them in memory
        for i in range(len(action_list)):
            action = action_list[i]
            reward = reward_list[i]
            x = state_list[i][1][action] # arm
            z = state_list[i][0] # user
            self.per_memory.push(z, x, action, reward)

        # Update model from sampled experiences
        if self.per_memory.size() < self.batch_size:
            return
        if self.per_memory.size() < self.batch_size*self.epochs:
            n_epoch = int(self.per_memory.size()/self.batch_size)
        else:
            n_epoch = self.epochs
        for _ in range(n_epoch):
            idxs, data_batch, is_weights = self.per_memory.sample(batch_size=self.batch_size)
            errors = []
            for data, is_weight in zip(data_batch, is_weights):
                reward = data.reward 
                x = data.arm_feat # arm
                z = data.user_feat # user
                x = np.concatenate((x,z), axis=0)
                x = torch.FloatTensor(x).reshape(1,-1).to(self.device)
                y = self.model(x)
                target = torch.FloatTensor([reward]).reshape(-1, 1)
                # print(y.shape, target.shape)

                loss = self.loss_fn(y, target) * is_weight
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                errors.append(reward - y.detach().cpu().item())
            self.per_memory.update(idxs, errors)

    def reset(self):
        self.per_memory = PER(
            memory_size=self.memory_size,
            a=self.a,
            beta=self.beta,
            beta_increment_per_sampling=self.beta_increment_per_sampling
        )
        self.model = LogisticRegression(input_size=self.arm_feat_dim+self.user_feat_dim,
                                        output_size=1)

    def default_prediction(self):
        return 