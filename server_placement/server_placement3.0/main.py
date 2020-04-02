import numpy as np
import random
import envirment
import torch
import ddpg_withLSTM1
# import ddpg
# from queue import Queue
from matplotlib import pyplot as plt
import config

# some parameters
MAX_EPISODES = config.MAX_EPISODES
SLOTNUM = config.SLOTNUM # the numbers of slots used
Bt_INDE_num = config.Bt_INDE_num

# about utilization ratio punishment
U_BEST = config.U_BEST # as the name say
PHI_1 = config.PHI_1 # means the punishment of utilization ratio when u<u_best
PHI_2 = config.PHI_2 #means the punishment of utilization ratio when u>u_best

# about INDE cost
ALPHA = config.ALPHA # means the cost of maintaining a basic station INDE
BETA = config.BETA # means the cost of maintaining a car INDE

# about centent loss punishment
OMEGA = config.OMEGA # means the punishment of centent loss

# Hyper Parameters
MEMORY_CAPACITY = config.MEMORY_CAPACITY

a_dim = config.a_dim
s_dim = config.s_dim

'''
main program
'''
# dqn = dqn.DQN()
ddpg = ddpg_withLSTM1.DDPG_withLSTM(a_dim, s_dim)
# ddpg = ddpg.DDPG(a_dim, s_dim)
env = envirment.Env(1001)
# retarder = Queue()

EPSILON = 0.3

a = np.zeros((1,a_dim))
for i in range(a_dim):
    if np.random.uniform()<0.2:
        a[0,i] = 1
a = np.ravel(a)

a_last = np.ravel(np.zeros((1,a_dim)))
s = np.ravel(np.zeros((1,s_dim)))
s_ = np.ravel(np.zeros((1,s_dim)))
s = env.update(a, a_last, 0, 0)
s_ = s
a_last = a
s = np.transpose(s)
s_ = np.transpose(s_)
plt.ion()

for i_episode in range(MAX_EPISODES):

    SystemPerformance = []
    L1_Car_Num_line = np.zeros((SLOTNUM,2))
    Reward_line = np.zeros((SLOTNUM,))
    Avg_Delay_line = np.zeros((SLOTNUM,))

    print('i_episode = '+str(i_episode))
    if i_episode == 0:
        spf = open(config.result_Location+'SystemPerformance/SystemPerformance.txt', 'w')
    else:
        spf = open(config.result_Location+'SystemPerformance/SystemPerformance.txt', 'a')
    spf.write('i_episode = '+str(i_episode)+'\n')





    for t in range(1, SLOTNUM):

        if i_episode == 0 and SLOTNUM>=50:
            continue

        # aa = ddpg.choose_action_2(s, EPSILON)
        # aa = ddpg.choose_action_0(s)
        # if i_episode<=12:
        #     aa = ddpg.choose_action_2(s, EPSILON)
        # else:
        #     aa = ddpg.choose_action_0(s)
        #     aa = aa.unsqueeze(0)
        if t%200 == 0:
            aa = ddpg.choose_action_0(s)
        else:
            aa = ddpg.choose_action_2(s, EPSILON)
        # print(t)
        # aa = aa.numpy()
        a = aa[0,:].numpy()
        # print(a)
        for i in range(a_dim):

            if a[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0
        # a = aa[0,:]
        # aa = torch.from_numpy(aa)
        s_ = env.update(a, a_last, i_episode, t)
        
        L1_Delay = env.Get_L1_Car_Delay()
        L1_Delay_sum = 0
        L1_Delay_count = 0
        for i in range(a_dim):
            if L1_Delay[i] != 0:
                L1_Delay_sum = L1_Delay_sum+L1_Delay[i]
                L1_Delay_count = L1_Delay_count+1
        if L1_Delay_count != 0:
            Avg_Delay_line[t] = L1_Delay_sum/L1_Delay_count
        # calculate INDE cost
        # cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])
        cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])

        # calculate utilization ratio reward
        reward_utilization = 0
        for i in range(a_dim):
            if s_[i]<=0.6:
                reward_utilization += (1/0.6)*s_[i]
            elif s_[i]<=1:
                reward_utilization += 1
            else:
                reward_utilization += (-s_[i]*s_[i]+s_[i]+1)
        # reward_utilization = np.sum(s_[0:a_dim]) 
        # panalty of overload
        # puni_overload = 0
        # for i in range(a_dim):
        #     if s_[i]>=1:
        #         puni_overload += 1
        L1_Car_Num_line[t,:] = env.Get_L1_Car_Num()
        car_num = (env.Print_System_Performance())[0]

        uti = np.squeeze(env.load_rate)
        open_uti_rate_sum = 0
        ave_open_uti_rate = 0
        for i in range(a_dim):
            if a[i] == 1:
                open_uti_rate_sum += uti[i]
        if np.sum(a)!= 0:
            ave_open_uti_rate = open_uti_rate_sum/np.sum(a)

        connect_rate = (env.Print_System_Performance())[1]

        # calculate total r
        # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
        r = 6*reward_utilization-cost_open
        # if np.sum(a) <= 5:
        #     r = r-9999
        r = np.array([r])
        Reward_line[t] = r
        s_ = np.transpose(s_)
        
        SystemPerformance.append([car_num, np.sum(a), connect_rate, ave_open_uti_rate, np.max(uti), round(float(r),1)])
        # SystemPerformance.append([np.sum(a), env.Print_System_Performance(),round(float(cost_open),1), round(float(puni_utilization),1), round(float(puni_contentloss),1)])
        
        if t%200 == 0:
            # print(t)
            SystemPerformance = np.array(SystemPerformance)
            print('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
            spf.write('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
            SystemPerformance = []
        
        # print(aa)
        # print(s)
        # print(aa)
        # print(r)
        # print(s_)
        ddpg.store_transition(s, aa, r, s_)
        if ddpg.pointer > MEMORY_CAPACITY and ddpg.pointer%32==0:
            ddpg.learn()
            
        s = s_
        # ep_r += r
        a_last = a

    # draw number of car and INDE,draw reward
    if i_episode == 0:
        continue
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0], color='blue', label='Car number')
    plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0]/config.L1_INDE_CAPACITY, color='yellow', label='ideal INDE number')
    plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,1], color='red', label='INDE number')
    plt.subplot(2,1,2)
    plt.plot(range(1,SLOTNUM), Reward_line[1:], color='green', label='Reward')
    # plt.show()
    plt.savefig('result/car_L1_reward_figure/i_episode_'+str(i_episode)+'.png')
    # plt.pause(1)

    dict_car_num = {}
    for i in range(1, SLOTNUM):
        if L1_Car_Num_line[i,0] in dict_car_num.keys():
            (dict_car_num[L1_Car_Num_line[i,0]])[0].append(Reward_line[i]) # Reward
            (dict_car_num[L1_Car_Num_line[i,0]])[1].append(L1_Car_Num_line[i,1]) # INDE number
            (dict_car_num[L1_Car_Num_line[i,0]])[2].append(Avg_Delay_line[i])
        else:
            dict_car_num.update({L1_Car_Num_line[i,0]:[[Reward_line[i]], [L1_Car_Num_line[i,1]], [Avg_Delay_line[i]]]})

    car_num_list = list(dict_car_num.keys())
    # print(car_num_list)
    # print(car_num_list)
    Reward_CarNum_list = np.zeros((len(car_num_list),))
    L1Num_CarNum_list = np.zeros((len(car_num_list),))
    Delay_CarNum_list = np.zeros((len(car_num_list),))
    for i in range(len(car_num_list)):
        Reward_CarNum_list[i] = sum((dict_car_num[car_num_list[i]])[0])/len((dict_car_num[car_num_list[i]])[0])
        L1Num_CarNum_list[i] = sum((dict_car_num[car_num_list[i]])[1])/len((dict_car_num[car_num_list[i]])[1])
        Delay_CarNum_list[i] = sum((dict_car_num[car_num_list[i]])[2])/len((dict_car_num[car_num_list[i]])[2])

    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(car_num_list, Reward_CarNum_list, color='green')
    plt.subplot(3,1,2)
    plt.plot(car_num_list, L1Num_CarNum_list, color='red')
    plt.subplot(3,1,3)
    plt.plot(car_num_list, Delay_CarNum_list, color='m')
    plt.savefig('result/car_L1_reward_figure/INDE-Car_i_episode_'+str(i_episode)+'.png')
    # EPSILON = EPSILON*0.8 


    env.Render()
    spf.close()

del env

# Date = 1005
# env = envirment.Env(Date)
# # retarder = Queue()

# EPSILON = 0.95

# a = np.zeros((1,a_dim))
# for i in range(a_dim):
#     if np.random.uniform()<0.2:
#         a[0,i] = 1
# a = np.ravel(a)
# a_last = np.ravel(np.zeros((1,a_dim)))
# s = np.ravel(np.zeros((1,s_dim)))
# s_ = np.ravel(np.zeros((1,s_dim)))
# s = env.update(a, a_last, 0, 0)
# s_ = s
# a_last = a
# s = np.transpose(s)
# s_ = np.transpose(s_)

# for test_episode in range(3):

#     SystemPerformance = []
#     L1_Car_Num_line = np.zeros((SLOTNUM,2))
#     Reward_line = np.zeros((SLOTNUM,))
#     print('test_episode = '+str(test_episode))
#     if test_episode == 0:
#         spf = open(config.result_Location+'SystemPerformance/test_episode_SystemPerformance.txt', 'w')
#     else:
#         spf = open(config.result_Location+'SystemPerformance/test_episode_SystemPerformance.txt', 'a')
#     spf.write('test_episode = '+str(test_episode)+'\n')
#     for t in range(1,SLOTNUM):

#         if test_episode == 0 and SLOTNUM>=50:
#             continue
#         if test_episode == 1:
#             aa = ddpg.choose_action_2(s, EPSILON)
#         if test_episode == 2:
#             aa = ddpg.choose_action_0(s)
#             aa = aa.unsqueeze(0)     
#         # if i_episode<=12:
#         #     aa = ddpg.choose_action_2(s, EPSILON)
#         # else:
#         #     aa = ddpg.choose_action_0(s)
#         #     aa = aa.unsqueeze(0)
#         # aa = ddpg.choose_action_0(s)
#         # aa = aa.unsqueeze(0)
#         # print(t)
#         # aa = aa.numpy()
#         a = aa[0,:].numpy()
#         # print(a)
#         for i in range(a_dim):

#             if a[i] >= 0.5:
#                 a[i] = 1
#             else:
#                 a[i] = 0
#         # a = aa[0,:]
#         # aa = torch.from_numpy(aa)
#         s_ = env.update(a, a_last, i_episode, t+(Date-1001)*SLOTNUM)
#         L1_Car_Num_line[t,:] = env.Get_L1_Car_Num()
#         # calculate INDE cost
#         # cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])
#         cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])

#         # calculate utilization ratio reward
#         reward_utilization = 0
#         for i in range(a_dim):
#             if s_[i]<=0.6:
#                 reward_utilization += (1/0.6)*s_[i]
#             elif s_[i]<=1:
#                 reward_utilization += 1
#             else:
#                 reward_utilization += (-s_[i]*s_[i]+s_[i]+1)
#         # reward_utilization = np.sum(s_[0:a_dim])

#         # panalty of overload
#         # puni_overload = 0
#         # for i in range(a_dim):
#         #     if s_[i]>=1:
#         #         puni_overload += 1

#         car_num = (env.Print_System_Performance())[0]

#         uti = np.squeeze(env.load_rate)
#         open_uti_rate_sum = 0
#         ave_open_uti_rate = 0
#         for i in range(a_dim):
#             if a[i] == 1:
#                 open_uti_rate_sum += uti[i]
#         if np.sum(a)!= 0:
#             ave_open_uti_rate = open_uti_rate_sum/np.sum(a)

#         connect_rate = (env.Print_System_Performance())[1]

#         # calculate total r
#         # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
#         r = 6*reward_utilization-cost_open
#         r = np.array([r])
#         Reward_line[t] = r
#         s_ = np.transpose(s_)
        
#         SystemPerformance.append([car_num, np.sum(a), connect_rate, ave_open_uti_rate, np.max(uti), round(float(r),1)])
#         # SystemPerformance.append([np.sum(a), env.Print_System_Performance(),round(float(cost_open),1), round(float(puni_utilization),1), round(float(puni_contentloss),1)])
#         if t%200 == 0:
#             # print(t)
#             SystemPerformance = np.array(SystemPerformance)
#             print('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
#             spf.write('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
#             SystemPerformance = []
#         # print(aa)
#         # print(s)
#         # print(aa)
#         # print(r)
#         # print(s_)
#         # ddpg.store_transition(s, aa, r, s_)
#         # if ddpg.pointer > MEMORY_CAPACITY:
#         #     ddpg.learn()
            
#         s = s_
#         # ep_r += r
#         a_last = a
#     # draw number of car and INDE,draw reward
#     plt.clf()
#     plt.subplot(2,1,1)
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0], color='blue', label='Car number')
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0]/config.L1_INDE_CAPACITY, color='yellow', label='ideal INDE number')
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,1], color='red', label='INDE number')
#     plt.subplot(2,1,2)
#     plt.plot(range(1,SLOTNUM), Reward_line[1:], color='green', label='Reward')
#     # plt.show()
#     plt.savefig('result/car_L1_reward_figure/test_episode_'+str(test_episode)+'.png')
#     # plt.pause(1)

#     # EPSILON = EPSILON*0.8  
#     env.Render()
#     spf.close()




























# env.Restart_With_A_New_Day(1002)

# a = np.zeros((1,a_dim))
# for i in range(a_dim):
#     if np.random.uniform()<0.2:
#         a[0,i] = 1
# a = np.ravel(a)
# a_last = np.ravel(np.zeros((1,a_dim)))
# s = np.ravel(np.zeros((1,s_dim)))
# s_ = np.ravel(np.zeros((1,s_dim)))
# s = env.update(a, a_last, 0)
# s_ = s
# a_last = a
# s = np.transpose(s)
# s_ = np.transpose(s_)

# for test_episode in range(3):
#     SystemPerformance = []
#     L1_Car_Num_line = np.zeros((SLOTNUM,2))
#     Reward_line = np.zeros((SLOTNUM,))
#     print('test_episode = '+str(test_episode))
#     if test_episode == 0:
#         spf = open(config.result_Location+'SystemPerformance/test_episode_SystemPerformance.txt', 'w')
#     else:
#         spf = open(config.result_Location+'SystemPerformance/test_episode_SystemPerformance.txt', 'a')
#     spf.write('test_episode = '+str(test_episode)+'\n')
#     for t in range(0,SLOTNUM):

#         if test_episode == 0 and SLOTNUM>=50:
#             continue
#         if test_episode == 1:
#             aa = ddpg.choose_action_2(s, EPSILON)
#         if test_episode == 2:
#             aa = ddpg.choose_action_0(s)
#             aa = aa.unsqueeze(0)            

#         a = aa[0,:].numpy()
#         # print(a)
#         for i in range(a_dim):

#             if a[i] >= 0.5:
#                 a[i] = 1
#             else:
#                 a[i] = 0
#         if test_episode != 0:
#             print(t)
#         s_ = env.update(a, a_last, (t+SLOTNUM))
#         L1_Car_Num_line[t,:] = env.Get_L1_Car_Num()

#         cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])
#         reward_utilization = np.sum(s_[0:a_dim])

#         car_num = (env.Print_System_Performance())[0]

#         uti = np.squeeze(env.load_rate)
#         open_uti_rate_sum = 0
#         ave_open_uti_rate = 0
#         for i in range(a_dim):
#             if a[i] == 1:
#                 open_uti_rate_sum += uti[i]
#         if np.sum(a)!= 0:
#             ave_open_uti_rate = open_uti_rate_sum/np.sum(a)

#         connect_rate = (env.Print_System_Performance())[1]

#         # calculate total r
#         # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
#         r = 6*reward_utilization-cost_open
#         r = np.array([[r]])
#         Reward_line[t] = r
#         s_ = np.transpose(s_)
        
#         SystemPerformance.append([car_num, np.sum(a), connect_rate, ave_open_uti_rate, np.max(uti), round(float(r),1)])
#         # SystemPerformance.append([np.sum(a), env.Print_System_Performance(),round(float(cost_open),1), round(float(puni_utilization),1), round(float(puni_contentloss),1)])
#         if t%200 == 0:
#             # print(t)
#             SystemPerformance = np.array(SystemPerformance)
#             print('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
#             spf.write('t = '+str(t-200)+'-'+str(t)+'   '+str(np.mean(SystemPerformance[0:200,:], axis = 0))+'\n')   
#             SystemPerformance = []
#         s = s_
#         # ep_r += r
#         a_last = a
#     # draw number of car and INDE,draw reward
#     plt.clf()
#     plt.subplot(2,1,1)
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0], color='blue', label='Car number')
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,0]/config.L1_INDE_CAPACITY, color='yellow', label='ideal INDE number')
#     plt.plot(range(1,SLOTNUM), L1_Car_Num_line[1:,1], color='red', label='INDE number')
#     plt.subplot(2,1,2)
#     plt.plot(range(1,SLOTNUM), Reward_line[1:], color='green', label='Reward')
#     # plt.show()
#     plt.savefig('result/car_L1_reward_figure/test_episode_'+str(test_episode)+'.png')
#     plt.pause(1)

#     # EPSILON = EPSILON*0.8  
#     env.Render()
#     spf.close()
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