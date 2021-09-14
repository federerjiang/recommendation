from porise.model.algorithms.mab import BetaThompsonSampling
from porise.model.algorithms.cmab import BayesianHybridLinUCBPER
from ..algo_base import AlgoBase

class HybridTSBHLUPER(AlgoBase):
    def __init__(self, 
                n_arms, 
                alpha, 
                arm_feat_dim=0, 
                user_feat_dim=0, 
                return_list=True,
                memory_size=int(1e5),
                prio_a=0.6,
                prio_beta=0.4,
                prio_e=0.001,
                beta_increment_per_sampling=0.4e-6,
                batch_size=128,
                epochs=10,
                ):
        self.model_bhluper = BayesianHybridLinUCBPER(
                n_arms=n_arms,
                alpha=alpha,
                arm_feat_dim=arm_feat_dim,
                user_feat_dim=user_feat_dim,
                return_list=return_list,
                memory_size=memory_size,
                prio_a=prio_a,
                prio_beta=prio_beta,
                prio_e=prio_e,
                beta_increment_per_sampling=beta_increment_per_sampling,
                batch_size=batch_size,
                epochs=epochs,
                )
        self.model_ts = BetaThompsonSampling(
                n_arms=n_arms,
                return_list=return_list
                )

    def predict(self, state):
        if state[0]:
            return self.model_bhluper.predict(state)
        else:
            return self.model_ts.predict(state)


    def train(self, action_list, state_list, reward_list):
        self.model_ts.train(action_list, state_list, reward_list)
        new_action_list = []
        new_state_list = []
        new_reward_list = []
        for action, state, reward in zip(action_list, state_list, reward_list):
            if state[0]:
                new_action_list.append(action)
                new_state_list.append(state)
                new_reward_list.append(reward)
        self.model_bhluper.train(
            action_list=new_action_list,
            state_list=new_state_list,
            reward_list=new_reward_list
        )

    def reset(self):
        self.model_bhluper.reset()
        self.model_ts.reset()

    def default_prediction(self):
        return 