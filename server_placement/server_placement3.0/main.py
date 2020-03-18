import numpy as np
import random
import envirment
import torch
# import dqn
import ddpg_withLSTM


# some parameters
MAX_EPISODES = 15
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
MEMORY_CAPACITY = 50

a_dim = 1910
s_dim = 1910*3

'''
main program
'''
# dqn = dqn.DQN()
ddpg = ddpg_withLSTM.DDPG_withLSTM(a_dim, s_dim)
env = envirment.Env()

EPSILON = 1

a = np.ravel(np.ones((1,a_dim)))
a_last = np.ravel(np.zeros((1,a_dim)))
s = np.ravel(np.zeros((1,s_dim)))
s_ = np.ravel(np.zeros((1,s_dim)))
s = env.update(a, a_last, 0)
s_ = s
a_last = a
s = np.transpose(s)
s_ = np.transpose(s_)

for i_episode in range(MAX_EPISODES):

    SystemPerformance = []
    print(i_episode)

    for t in range(1,SLOTNUM):

        aa = ddpg.choose_action_2(s, EPSILON)

        # print(t)
        # aa = aa.numpy()
        a = aa[0,:].numpy()
        for i in range(a_dim):

            if a[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0
        # a = aa[0,:]
        # aa = torch.from_numpy(aa)
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
            if a_last[i] == 1 and a[i] == 0:
                puni_contentloss += OMEGA

        car_num = (env.Print_System_Performance())[0]
        connect_rate = (env.Print_System_Performance())[1]

        # calculate total r
        # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
        r = 400*connect_rate-cost_open
        r = np.array([r])
        s_ = np.transpose(s_)
        
        SystemPerformance.append([car_num, np.sum(a), connect_rate ,round(float(r),1)])
        # SystemPerformance.append([np.sum(a), env.Print_System_Performance(),round(float(cost_open),1), round(float(puni_utilization),1), round(float(puni_contentloss),1)])
        if t%199 == 0:
            SystemPerformance = np.array(SystemPerformance)
            print(SystemPerformance)
            if i_episode == 0:
                f = open('SystemPerformance/SystemPerformance.txt', 'w')
            else:
                f = open('SystemPerformance/SystemPerformance.txt', 'a')
            f.write('i_episode = '+str(i_episode)+'     '+str(np.mean(SystemPerformance[100:150,:], axis = 0))+'\n')
            f.close()
        ddpg.store_transition(s, aa, [r], s_)
        if ddpg.pointer > MEMORY_CAPACITY:
            ddpg.learn()
            
        s = s_
        # ep_r += r
        a_last = a
    EPSILON = EPSILON*0.99  
    env.Render()

'''
envirment test module
'''
# # create random actions which are used to test
# OPENRATE = 0.05
# aa = np.zeros((SLOTNUM,a_dim))
# for i in range(SLOTNUM):
#     for j in range(a_dim):
#         if random.random() <= OPENRATE:
#             aa[i,j] = 1
# aa = np.around(aa)
# SystemPerformance = []
# for i_episode in range(MAX_EPISODES):

#     a = np.ravel(np.ones((1,a_dim)))
#     a_last = np.ravel(np.zeros((1,a_dim)))
#     s = np.ravel(np.zeros((1,s_dim)))
#     s_ = np.ravel(np.zeros((1,s_dim)))

#     s = env.update(a, a_last, 0)
#     s_ = s
#     a_last = a
#     # ep_r = 0
#     s = np.transpose(s)
#     s_ = np.transpose(s_)

#     for t in range(1,SLOTNUM):
#         a = aa[t,:]
#         print(t)
#         # print('\n')
#         # print(a)
#         # a = aa[0,:].numpy()
#         # for i in range(a_dim):

#         #     if a[i] >= 0.5:
#         #         a[i] = 1
#         #     else:
#         #         a[i] = 0
#         # print(a)
#         s_ = env.update(a, a_last, t)

#         # calculate INDE cost
#         cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])

#         # calculate utilization ratio punishment
#         puni_utilization = 0
#         for i in range(s_dim):
#             if s_[i]<=U_BEST:
#                 puni_utilization += PHI_1*(U_BEST-s_[i])
#             else:
#                 puni_utilization += PHI_2*(s_[i]-U_BEST)

#         # calculate centent loss punishment
#         puni_contentloss = 0
#         for i in range(a_dim):
#             if a[i] == 1 and a[i] == 0:
#                 puni_contentloss += OMEGA

#         # calculate total r
#         r = -cost_open-puni_utilization-puni_contentloss
#         s_ = np.transpose(s_)
#         SystemPerformance.append([env.Print_System_Performance(),round(float(-r),1)])
#         # ddpg.store_transition(s, aa, r/10, s_)
#         if t%10 == 0:
#             print(SystemPerformance)
#         # if ddpg.pointer > MEMORY_CAPACITY:
#         #     dqn.learn()
            
#         s = s_
#         # ep_r += r
#         a_last = a