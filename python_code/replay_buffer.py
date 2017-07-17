""" 
Data structure for implementing experience replay

Author: Patrick Emami, Modified by: Sri Ramana
"""
from collections import deque
import random
import numpy as np
from copy import deepcopy

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, traj):
        if self.count < self.buffer_size:
            self.buffer.append(traj)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(traj)

    '''
    def makeHistory(self, traj):
        hist = {}
        for id, seq in traj:
            hist[id] = []
            oa[id] = []
            r[id] = []
            for j, val in enumerate(seq):
                o, a ,r = val
                oa = np.concatenate([o,a], axis =1)

    '''


    def MergeData(self, batch, next = False):
        new_batch = {}
        next_batch = {}
        seq = []
        for b in batch:
            cnt = 0
            for key in b:
                if cnt not in new_batch:
                    data = b[key]
                    new_batch[cnt] = deepcopy(data)
                    new_batch[cnt].pop(-1)
                    if next:
                        next_batch[cnt] = deepcopy(data)
                        next_batch[cnt].pop(-1)
                else:
                    data1 = deepcopy(b[key])
                    data2 = deepcopy(b[key])
                    data1.pop(-1)
                    new_batch[cnt] += data1

                    if next:
                        data2.pop(0)
                        next_batch[cnt] += deepcopy(data)
                print len(new_batch[cnt]), len(b[key])
                seq += np.arange(len(new_batch[cnt])).tolist()
                cnt += 1

        return new_batch, next_batch, seq

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        oa_batch, oa2_batch, seq = self.MergeData([_[0] for _ in batch], True)
        a_batch, _, _ =  self.MergeData([_[1] for _ in batch])
        r_batch, _, _ =  self.MergeData([_[2] for _ in batch])



        return oa_batch, a_batch, r_batch, oa2_batch, seq



        '''
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch
        '''
        #return batch


    def clear(self):
        self.deque.clear()
        self.count = 0


