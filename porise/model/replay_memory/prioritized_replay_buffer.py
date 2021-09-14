from .sum_tree import SumTree
import numpy as np 
import random
from collections import namedtuple


Transition = namedtuple('Transition', ('user_feat', 'arm_feat', 'action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.count = 0
        self.full = False

    def push(self, *args):
        """Saves a transition"""
        if self.full == False:
            self.count += 1
        if self.count == self.capacity:
            self.full = True
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Here will define oversampling method to handle imbalance click/non-click data
        """
        idxs = random.sample(range(int(self.count)), batch_size)
        batch = list(map(self.memory.__getitem__, idxs))
        return batch

    def __len__(self):
        return len(self.memory)


class PER(object):
    # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, memory_size=100000, a = 0.6, beta = 0.4, e = 0.0001, beta_increment_per_sampling = 0.4e-6):
        self.tree =  SumTree(memory_size)
        self.memory_size = memory_size
        self.prio_max = 0.1
        self.a = a
        self.beta = beta
        self.e = e
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
    def push(self, *args):
        """Saves a transition"""
        data = Transition(*args)
        p = (np.abs(self.prio_max) + self.e) ** self.a #  proportional priority
        self.tree.add(p, data)

    def sample(self, batch_size):
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling]) # max to 1
        idxs = []
        data_batch = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            data_batch.append(data)
            priorities.append(p)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        # is_weight = np.clip(is_weight, 0, 1)
        return idxs, data_batch, is_weight
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p) 
        
    def size(self):
        return self.tree.n_entries