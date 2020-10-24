# Libs used
import numpy as np
import random
# Modules used
import config

SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class Greedy(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim


    def choose_action(self, env, s, a_last):

        a = np.zeros((1, 2*self.a_dim))
        if np.sum(a_last) == 0:
            for i in range(self.a_dim):
                if random.random()< 0.2:
                    a[0,i] = 1
                    a[0,i+self.a_dim] = 0
                else:
                    a[0,i] = 0
                    a[0,i+self.a_dim] = 1
        else:
            for i in range(self.a_dim):
                if s[0,i] != 0:
                    a[0,i] = 1
                    a[0,i+self.a_dim] = 0
                    if s[0,i] >= 1:
                        neighbor = (env.list_INDE_object[i]).neighbor
                        for j in range(len(neighbor)):
                            closest_neighbor_id = int(neighbor[j,0])
                            if a_last[closest_neighbor_id] == 0:
                                if random.random()< 0.6:
                                    a[0,closest_neighbor_id] = 1
                                    a[0,closest_neighbor_id+self.a_dim] = 0
                                break
                elif a_last[i] == 1:
                    a[0,i] = 0
                    a[0,i+self.a_dim] = 1
        return a
