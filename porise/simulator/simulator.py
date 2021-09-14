import numpy as np 
from tqdm import tqdm 
from .plot import IpynPlot
import time 


class Simulator():
    """ Simulator
    """

    def __init__(self,
            model,
            env,
            train_every=int(1e2),
            throttle=int(1e2),
            memory_capacity=int(1e3),
            plot_every=int(1e4)
            ):
        self.model = model
        self.env = env 
        self.train_every = train_every
        self.throttle = throttle
        self.memory_capacity = memory_capacity
        self.max_steps = env.max_steps
        self.plot_every = plot_every
        self.plot = IpynPlot(plot_every=plot_every)
        self.times = []
        self.reset(train_every=train_every, memory_capacity=memory_capacity) 

    def reset_rewards(self):
        self.rewards = np.empty(self.max_steps)        

    def reset(self, 
            train_every=int(1e2),
            memory_capacity=int(1e3)):
        self.reset_rewards()
        self.env.reset()
        self.train_every = train_every
        self.memory_capacity = memory_capacity

    def run(self):
        """ Run an episode of bandit
        """
        postfix = {
            'total reward': 0.0
        }
        action_list = []
        feat_list = []
        reward_list = []
        state = self.env.get_user_state()
        with tqdm(total=self.max_steps-1, postfix=postfix) as pbar:
            for step in range(self.max_steps-1):
                assert len(action_list) <= self.memory_capacity
                assert len(feat_list) <= self.memory_capacity
                assert len(reward_list) <= self.memory_capacity

                action = self.model.predict(state)
                old_state = state 
                state, reward, done, info = self.env.step(action)
                
                action_list.append(action)
                feat_list.append(old_state)
                reward_list.append(reward)
                if step >= self.memory_capacity:
                    action_list.pop(0)
                    feat_list.pop(0)
                    reward_list.pop(0)
                
                self.rewards[step] = reward

                # update model
                if step % self.train_every == 0:
                    start = time.time()
                    self.model.train(action_list=action_list,
                                    state_list=feat_list,
                                    reward_list=reward_list)
                    end = time.time()
                    self.times.append(end-start)
                
                # log
                postfix['total reward'] += reward
                if step % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)
                if step % self.plot_every == 0:
                    self.plot.plot_process(self.rewards[0:step])
                

