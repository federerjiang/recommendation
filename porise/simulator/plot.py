import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import clear_output

class IpynPlot():
    """Plots logging data in jupyter notebook
    """

    def __init__(self, plot_every=1):
        self.plt = plt 
        self.clear_output = clear_output
        self.plot_every = plot_every
    
    def plot_process(self, rewards):
        self.clear_output()
        rewards = np.array(rewards)
        n_point = int(rewards.size/self.plot_every)
        acc_rewards = rewards.reshape((n_point, self.plot_every)).sum(axis=1)
        x = list(range(1, n_point+1))
        self.plt.plot(x, acc_rewards, label='Accumulated rewards', color='blue')
        self.plt.legend()
        self.plt.xlabel('epochs')
        self.plt.ylabel('loss')
        self.plt.show()



