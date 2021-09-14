import numpy as np 
import torch
import torch.nn as nn
from ..algo_base import AlgoBase
from porise.model.dnn.fc_model import FCModel
from porise.model.util import inv_sherman_morrison


class NeuralTS(AlgoBase):
    """
    Neural Thompson Sampling

    Parameters
    ----------
    n_arms: int
        Number of arms
    return_list: bool
        True: return the list of predicted logits of each arm
        False: return the index of the arm which has the largest logit

    References
    ----------
    Zhang, Weitong, Dongruo Zhou, Lihong Li, and Quanquan Gu. 
    "Neural Thompson Sampling."(2020).
    """
    def __init__(self, 
                n_arms,
                hidden_size=20,
                n_layers=2,
                learning_rate=0.01,
                epochs=1,
                use_cuda=False,
                p=0.0,
                arm_feat_dim=0, 
                user_feat_dim=0,
                out_size=1,
                memory_capacity=1e6,
                batch_size=128,
                confidence_multiplier=-1.0,
                return_list=True):
        super().__init__(n_arms)
        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers
        ## NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.use_cuda = use_cuda 
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        
        # drop rate
        self.p = p

        self.arm_feat_dim = arm_feat_dim
        self.user_feat_dim = user_feat_dim
        self.out_size = out_size

        # neural network
        self.model = FCModel(input_size=self.arm_feat_dim+self.user_feat_dim,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            out_size=self.out_size,
                            p=self.p,
                            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.memory_capacity = memory_capacity
        self.memory = ReplayMemory(self.memory_capacity)
        self.batch_size = batch_size
        self.confidence_multiplier = confidence_multiplier
        self.return_list = return_list
        self.approximator_dim = self.get_approximator_dim()
        self.reset_Ainv()
    
    def reset_Ainv(self):
        """Initialize n_arms square matrices representing the inverses
        of exploration bonus matrices.
        """
        self.Ainv = np.array(
            [
                np.eye(self.approximator_dim) for _ in range(self.n_arms)
            ]
        )
    
    def update_Ainv(self, arm, grad):
        self.Ainv[arm] = inv_sherman_morrison(
            grad,
            self.Ainv[arm]
        )

    def get_approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def predict(self, arm_feat_list):
        pred_ucb = []
        for i_arm in range(self.n_arms):
            x = torch.FloatTensor(arm_feat_list[i_arm]).reshape(1,-1).to(self.device)
            self.model.zero_grad()
            y = self.model(x)
            y.backward()
            # get grad for x
            grad = torch.cat([w.grad.detach().flatten() / np.sqrt(self.hidden_size) 
                for w in self.model.parameters() if w.requires_grad]).to(self.device)
            exploration_bouns = self.confidence_multiplier * np.sqrt(np.dot(grad, np.dot(self.Ainv[i_arm], grad.T)))
            self.update_Ainv(i_arm, grad)
            pred_ucb.append(np.random.normal(y.detach().numpy(), exploration_bouns, 1)[0])
        if self.return_list:
            return pred_ucb
        else:
            return int(np.argmax(pred_ucb))

    def train(self, action_list, arm_feat_list, reward_list):
        """
        Train neural network

        Parameters
        ----------
        action_list: [1, 2, 4] list 
            The arms which are seleted by the system
        reward_list: [1, 0, 1] list
            The corresponding rewards of the selected arms
        """
        assert len(action_list) == len(reward_list)
        assert len(action_list) == len(arm_feat_list)
        
        # collect the experiences, and store them in memory
        for reward, state_t in zip(reward_list, arm_feat_list):
            reward = torch.tensor([reward], dtype=torch.float)
            state_t = torch.tensor([state_t], dtype=torch.float)
            self.memory.push(state_t, reward)

        if len(self.memory) < self.batch_size:
            return
        # train model
        self.model.train()
        train_count = len(action_list)/self.batch_size * self.epochs
        for _ in range(train_count):
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            reward_batch = torch.cat(batch.reward).to(self.device)
            reward_pred = self.model.forward(state_batch).squeeze()
            loss = self.criterion(reward_batch, reward_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def reset(self):
        # neural network
        self.model = FCModel(input_size=self.arm_feat_dim+self.user_feat_dim,
                            hidden_size=self.hidden_size,
                            n_layers=self.n_layers,
                            out_size=self.out_size,
                            p=self.p,
                            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_capacity) 
        self.reset_Ainv()

    def default_prediction(self):
        return 