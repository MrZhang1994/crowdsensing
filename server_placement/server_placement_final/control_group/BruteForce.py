# Libs used
import numpy as np
import random
import time
import copy
import gc

# Modules used
import config


# some parameters
MAX_EPISODES = config.MAX_EPISODES
SLOTNUM = config.SLOTNUM # the numbers of slots used
Bt_INDE_num = config.Bt_INDE_num

# about INDE cost
ALPHA = config.ALPHA # means the cost of maintaining a basic station INDE
BETA = config.BETA # means the cost of maintaining a car INDE

# about centent loss punishment
OMEGA = config.OMEGA # means the punishment of centent loss

# Hyper Parameters
MEMORY_CAPACITY = config.MEMORY_CAPACITY

a_dim = config.a_dim
s_dim = config.s_dim


def Calculate_Reward(a, uti):
    # calculate INDE cost
    # cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])
    cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])

    # calculate utilization ratio reward
    reward_utilization = 0
    for i in range(a_dim):
        if uti[i]<=0.6:
            reward_utilization += (1/0.6)*uti[i]
        elif uti[i]<=1:
            reward_utilization += 1
        else:
            reward_utilization += (-uti[i]*uti[i]+uti[i]+1)
    # calculate total r
    # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
    r = 6*reward_utilization-cost_open
    return r

class BruteForce(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim


    def choose_action(self, env, a_last, i_episode, t, Date, SLOTNUM):
        action = np.zeros((1, self.a_dim))
        max_r = -9999999
        t_start = time.time()
        while(1):
            a = np.zeros((1, self.a_dim))
            ep = random.random()
            if np.sum(a_last) == 0:
                for i in range(self.a_dim):
                    if random.random()< ep:
                        a[0,i] = 1
            r = self.Calculate_Reward_in_Other_Action(env, a[0,:], a_last, i_episode, t, Date, SLOTNUM)
            if r> max_r:
                action = a
                max_r = r

            if time.time()-t_start>10:
                break
        return action

    def Calculate_Reward_in_Other_Action(self, env, other_action, a_last, i_episode, t, Date, SLOTNUM):
        temp_env = copy.deepcopy(env)
        temp_s_ = temp_env.update_only(other_action, a_last, i_episode, t+(Date-1001)*SLOTNUM)
        temp_r = Calculate_Reward(other_action, temp_s_)
        del temp_env
        gc.collect()
        return temp_r    