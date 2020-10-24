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


        self.s_last_2 = np.zeros((1, a_dim))
        self.s_last_3 = np.zeros((1, a_dim))      
        self.a_last_2 = np.zeros(a_dim,)
        self.a_last_3 = np.zeros(a_dim,)

    def choose_action(self, env, s, a_last):
        a = np.zeros((1, self.a_dim))
        if np.sum(a_last) == 0:
            for i in range(self.a_dim):
                if random.random()< 0.2:
                    a[0,i] = random.uniform(0.6, 0.9)
                else:
                    a[0,i] = random.uniform(0.1, 0.4)
        else:
            for i in range(self.a_dim):
                if s[0,i] != 0:
                    a[0,i] = random.uniform(0.6, 0.9)
                    if s[0,i] >= 1:
                        neighbor = (env.list_INDE_object[i]).neighbor
                        for j in range(len(neighbor)):
                            closest_neighbor_id = int(neighbor[j,0])
                            if a_last[closest_neighbor_id] == 0:
                                if random.random()< 0.6:
                                    a[0,closest_neighbor_id] = random.uniform(0.6, 0.9)
                                break
                else:
                    if a_last[i] == 0:
                        a[0,i] = random.uniform(0.1, 0.4)
                    else:
                        if self.a_last_2[i]==1 and self.s_last_2[0,i]==0 and self.a_last_3[i]==1 and self.s_last_3[0,i]==0:
                            a[0,i] = random.uniform(0.1, 0.4)
                        else:
                            a[0,i] = random.uniform(0.6, 0.9)

        self.a_last_3 = self.a_last_2
        self.a_last_2 = a_last
        self.s_last_3 = self.s_last_2   
        self.s_last_2 = s

        return self.a
