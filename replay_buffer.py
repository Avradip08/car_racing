""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import pickle

#from frame_encoder import crop
import threading

_LOCK = threading.Lock()


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, experience):
        _LOCK.acquire()
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        _LOCK.release()

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        td_batch = np.array([_[3] for _ in batch])

        return s_batch, a_batch, r_batch, td_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

    def save(self, file_name):
        """
        save itself
        """
        print "==================================="
        print "Buffer Size : "+str(self.buffer_size)
        print "Count : "+str(self.count)
        print "==================================="
        f = open(file_name, 'wb')
        pickle.dump(self, f)
        f.close()


def load_rb(file_name):
    f = open(file_name, 'rb')
    c = pickle.load(f)
    print "==================================="
    print "Buffer Size : "+str(c.buffer_size)
    print "Count : "+str(c.count)
    print "==================================="
    f.close()
    return c
