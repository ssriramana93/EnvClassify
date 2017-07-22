""" 
Data structure for implementing experience replay

Author: Patrick Emami, Modified by: Sri Ramana
"""
from collections import deque
import random
import numpy as np
from copy import deepcopy

class ReplayBuffer(object):

    def __init__(self, buffer_size, min_hlen = 0, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.min_hlen = min_hlen
        self.count = 0
        self.ncount = 0
        self.buffer = deque()
        self.oa_batch = {}
        self.oa2_batch = {}
        self.seq = []
        self.a_batch = {}
        self.r_batch = {}
        random.seed(random_seed)

    def add(self, traj):
        if self.count < self.buffer_size:
            self.AppendData(traj)


#            self.buffer.append(traj)
#            self.count += 1
        else:
            self.RemoveFirst(self.ncount)
            self.AppendData(traj)
        #    self.buffer.popleft()
        #    self.buffer.append(traj)

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
    def RemoveFirst(self, count):
        if not self.oa_batch:
            return
        else:
            for key in self.oa_batch:
                del self.oa_batch[key][:count]
                del self.oa2_batch[key][:count]
                del self.a_batch[key][:count]
                del self.r_batch[key][:count]
            del self.seq[:count]



    def AppendData(self, traj):
        #oa_batch, oa2_batch, seq = self.MergeData([_[0] for _ in traj], True)
        #a_batch, _, _ = self.MergeData([_[1] for _ in traj])
        #r_batch, _, _ = self.MergeData([_[2] for _ in traj])
        oa_batch, a_batch, r_batch = traj
        oa2_batch = {}
        length = 0
        cnt = 0
        oa_copy = {}
        for key in oa_batch:
            data = oa_batch[key]
            oa2_batch[cnt] = deepcopy(data)[self.min_hlen + 1:]
            oa_copy[cnt] = deepcopy(data)[self.min_hlen:-1]
            length = len(data) - 1
            cnt += 1
        seq = (np.arange(self.min_hlen, length) + 1).tolist()


        '''
        if not self.oa_batch:
            self.oa_batch = oa_batch
            self.oa2_batch = oa2_batch
            self.a_batch = a_batch
            self.r_batch = r_batch
            self.seq = seq
        else:
        '''
        cnt = 0
        for key in oa_batch:
                if cnt not in self.oa_batch:
                    self.oa_batch[cnt] = []
                    self.oa2_batch[cnt] = []
                    self.a_batch[cnt] = []
                    self.r_batch[cnt] = []
                self.oa_batch[cnt] += oa_copy[cnt]
                self.oa2_batch[cnt] += oa2_batch[cnt]
                self.a_batch[cnt] += a_batch[key][self.min_hlen:-1]
                self.r_batch[cnt] += r_batch[key][self.min_hlen:-1]
                cnt += 1
        self.seq += seq

        self.count += len(seq)
        self.ncount = len(seq)

    def MergeData(self, batch, next = False):
        new_batch = {}
        next_batch = {}
        seq = []
        length = None
        for b in batch:
            cnt = 0
            for key in b:
                length = len(b[key])
                if cnt not in new_batch:

                    new_batch[cnt] = deepcopy(b[key])[:-1]
                    #new_batch[cnt].pop(-1)
                    if next:
                        next_batch[cnt] = deepcopy(b[key])[1:]
                        #next_batch[cnt].pop(-1)
                else:
                    #data1 = deepcopy(b[key])
                    #data2 = deepcopy(b[key])
                    #data1.pop(-1)
                    new_batch[cnt] += deepcopy(b[key])[:-1]

                    if next:
                        #data2.pop(0)
                        next_batch[cnt] += deepcopy(b[key])[1:]
                cnt += 1

        seq = np.arange(length).tolist()*len(batch)
        return new_batch, next_batch, seq

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            #batch = random.sample(self.buffer, self.count)
            index = np.random.choice(self.count, self.count)
        else:
            #batch = random.sample(self.buffer, batch_size)
            index = np.random.choice(self.count, batch_size)
        index = index.tolist()
        oa_batch = {}
        a_batch = {}
        r_batch = {}
        oa2_batch = {}
        for key in self.oa_batch:
            oa_batch[key] = [self.oa_batch[key][i] for i in index]
            a_batch[key] = [self.a_batch[key][i] for i in index]
            r_batch[key] = [self.r_batch[key][i] for i in index]
            oa2_batch[key] = [self.oa2_batch[key][i] for i in index]
        print len(self.seq), self.count
        seq = [self.seq[i] for i in index]

        return oa_batch, a_batch, r_batch, oa2_batch, seq

        '''
        oa_batch, oa2_batch, seq = self.MergeData([_[0] for _ in batch], True)
        a_batch, _, _ =  self.MergeData([_[1] for _ in batch])
        r_batch, _, _ =  self.MergeData([_[2] for _ in batch])


        return oa_batch, a_batch, r_batch, oa2_batch, seq
        '''


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


