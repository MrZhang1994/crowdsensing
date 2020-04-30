# Libs used
import numpy as np
import random
# Modules used
import config

SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class Random(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim
        

    def choose_action(self):
        action = np.zeros((1, self.a_dim))
        for i in range(self.a_dim):
            if random.random() <= 0.2:
                action[0,i] = 1  
            else:
                action[0,i] = 0
        return action
