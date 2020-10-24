# Libs used
import numpy as np
import random
# Modules used
import config

SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class ParaTable(object):
    def __init__(self, a_dim, s_dim, SLOTNUM):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.Table = np.loadtxt('Table.txt')
        self.LR = 0.1

    def choose_action(self, t):

        action = np.zeros((1, self.a_dim))

        for i in range(self.a_dim):
            if random.random() <= self.Table[t,i]:
                action[0,i] = 1  
            else:
                action[0,i] = 0

        return action

    def learn(self, t, a, s_):
        for i in range(self.a_dim):
            if a[0,i] == 1 and s_[0,i] <= 0.2:
                if self.Table[t,i]>0:
                    self.Table[t,i] -= self.LR
                else:
                    self.Table[t,i] = 0
            if a[0,i] == 1 and s_[0,i] > 0:
                if s_[0,i] >= 0.8:
                    for j in range(-2,3):
                        if (i+j) in range(self.a_dim):
                            if self.Table[t,i+j] < 1:
                                self.Table[t,i+j] += self.LR/(1+abs(j))
                            else:
                                self.Table[t,i+j] = 1
                        if (t+j) in range(SLOTNUM):
                            if self.Table[t+j,i] < 1:
                                self.Table[t+j,i] += self.LR/(1+abs(j))
                            else:
                                self.Table[t+j,i] = 1
                else:
                    for j in range(-2,3):
                        if (t+j) in range(SLOTNUM):
                            if self.Table[t+j,i] < 1:
                                self.Table[t+j,i] += self.LR/(1+abs(j))
                            else:
                                self.Table[t+j,i] = 1                    
