import numpy as np
import random
import envirment
import torch
# import dqn
import ddpg


# some parameters
MAX_EPISODES = 1
SLOTNUM = 200 # the numbers of slots used
Bt_INDE_num = 1910

# about utilization ratio punishment
U_BEST = 0.7 # as the name say
PHI_1 = 1 # means the punishment of utilization ratio when u<u_best
PHI_2 = 1 #means the punishment of utilization ratio when u>u_best

# about INDE cost
ALPHA = 1 # means the cost of maintaining a basic station INDE
BETA = 1 # means the cost of maintaining a car INDE

# about centent loss punishment
OMEGA = 1 # means the punishment of centent loss

# Hyper Parameters
MEMORY_CAPACITY = 2000

a_dim = 1989
s_dim = 1989

        
'''
envirment test module
'''
# create random actions which are used to test
OPENRATE = 1
aa = np.zeros((SLOTNUM,a_dim))
for i in range(SLOTNUM):
    for j in range(a_dim):
        if random.random() <= OPENRATE:
            aa[i,j] = 1
# aa = np.around(aa)


'''
main program
'''
# dqn = dqn.DQN()
ddpg = ddpg.DDPG(a_dim, s_dim)
env = envirment.Env()

for i_episode in range(MAX_EPISODES):

    a = np.ravel(np.ones((1,a_dim)))
    a_last = np.ravel(np.zeros((1,a_dim)))
    s = np.ravel(np.zeros((1,s_dim)))
    s_ = np.ravel(np.zeros((1,s_dim)))

    s = env.update(a, a_last, 0)
    s_ = s
    a_last = a
    # ep_r = 0
    s = np.transpose(s)
    s_ = np.transpose(s_)

    for t in range(1,SLOTNUM):
        aa = ddpg.choose_action(s)
        # a = aa[t,:]
        print(t)
        # print('\n')
        # print(a)
        a = aa[0,:].numpy()
        for i in range(a_dim):

            if a[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0

        s_ = env.update(a, a_last, t)

        # calculate INDE cost
        cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])

        # calculate utilization ratio punishment
        puni_utilization = 0
        for i in range(s_dim):
            if s_[i]<=U_BEST:
                puni_utilization += PHI_1*(U_BEST-s_[i])
            else:
                puni_utilization += PHI_2*(s_[i]-U_BEST)

        # calculate centent loss punishment
        puni_contentloss = 0
        for i in range(a_dim):
            if a[i] == 1 and a[i] == 0:
                puni_contentloss += OMEGA

        # calculate total r
        r = -cost_open-puni_utilization-puni_contentloss
        s_ = np.transpose(s_)
        
        ddpg.store_transition(s, aa, r/10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            dqn.learn()
            
        s = s_
        # ep_r += r
        a_last = a
