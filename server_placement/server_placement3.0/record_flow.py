# Libs used
import numpy as np
from matplotlib import pyplot as plt
import torch
from datetime import datetime
import os
import xlwt

# Modules used
import envirment
import output
import PDDPG
import control_group.GreedyPolicy
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




dt=datetime.now() #创建一个datetime类对象
Time = str(dt.month).zfill(2)+str(dt.day).zfill(2)+'_'+str(dt.hour).zfill(2)+str(dt.minute).zfill(2)
del dt
path = 'result/'+Time

if os.path.exists(path) == False:
    os.makedirs(path);
    os.makedirs(path+'/car_L1_reward_figure')
    os.makedirs(path+'/AdaptSpeed')
    os.makedirs(path+'/ConvergenceRate')
    os.makedirs(path+'/log')
    os.makedirs(path+'/result_images')
    os.makedirs(path+'/SystemPerformance')

Date = 1001
'''
main program
'''
# dqn = dqn.DQN()
Agent = control_group.GreedyPolicy.Greedy(a_dim)
# ddpg = PDDPG.PDDPG(a_dim, s_dim)
# ddpg = ddpg.DDPG(a_dim, s_dim)
env = envirment.Env(Date, path)
# wb = xlwt.Workbook(encoding="utf-8")
# ws = wb.add_sheet('flow_record', cell_overwrite_ok=True)
# retarder = Queue()
# sys_per = output.SystemPerformance(a_dim)
# line_fig = output.LineFigures(a_dim)
# conv_rate = output.ConvergenceRate(MAX_EPISODES, SLOTNUM, a_dim, s_dim)
# adapt_speed = output.AdaptSpeed(SLOTNUM, a_dim)

flow_record = np.zeros((SLOTNUM, a_dim))

EPSILON = 0.95

a = np.zeros((1,a_dim))
for i in range(a_dim):
    if np.random.uniform()<0.2:
        a[0,i] = 1
a = np.ravel(a)

a_last = np.ravel(np.zeros((1,a_dim)))
s = np.ravel(np.zeros((1,s_dim)))
s_ = np.ravel(np.zeros((1,s_dim)))
s = env.update(a, a_last, 0, 0)[0]
s_ = s
a_last = a
s = np.transpose(s)
s_ = np.transpose(s_)

for i_episode in range(1):

    print('i_episode = '+str(i_episode))
    # sys_per.reset(Date, i_episode, 0)
    # line_fig.reset()
    # adapt_speed.reset(SLOTNUM, a_dim)
    # conv_rate.reset(Date, path)

    for t in range(1, SLOTNUM):

        aa = Agent.choose_action(env, s, a_last)
        a = aa[0,:]


        # open_state = env.Report_open_close()
        # for i in range(a_dim):
        #     if a_last[i]!=open_state[i]:
        #         print('1_ERROR!!!!!')
        #         break

        # if i_episode%3==0 and t%100==0:
        #     result_list = adapt_speed.Calculate_AdaptSpeed(env, a_last, i_episode, t, Date, SLOTNUM)
        #     reward_list = result_list[0] 
        #     disconnect_rate_list = result_list[1]
        #     Delay_Outage_Rate_list = result_list[2]

        result = env.update(a, a_last, i_episode, t+(Date-1001)*SLOTNUM)
        s_ = result[0]
        flow_record[t,:] = result[1]
        # for i in range(a_dim):
        #     ws.write(t, i, flow_record[t,i])
        # r = output.Calculate_Reward(env, a, s_)

        # if i_episode%3==0 and t%100==0:
        #     reward_list.append(float(r))
        #     disconnect_rate = 1-(env.Print_System_Performance())[1]
        #     L1_Delay = env.Get_L1_Car_Delay()
        #     L1_Delay_count = 0
        #     L1_Delay_outrage_count = 0
        #     Delay_Outage_Rate = 0
        #     for i in range(a_dim):
        #         if L1_Delay[i] != 0:
        #             L1_Delay_count = L1_Delay_count+1
        #         if L1_Delay[i]>50:
        #             L1_Delay_outrage_count = L1_Delay_outrage_count+1
        #     if L1_Delay_count != 0:
        #         Delay_Outage_Rate = L1_Delay_outrage_count/L1_Delay_count
        #     disconnect_rate_list.append(float(disconnect_rate))
        #     Delay_Outage_Rate_list.append(float(Delay_Outage_Rate))
        #     sys_per.update_AdaptSpeed([reward_list, disconnect_rate_list, Delay_Outage_Rate_list], t)
    
        s_ = np.transpose(s_)

        # line_fig.update(env, r, t)
        # sys_per.update(env, a, r, t, path)
        # conv_rate.store_reward(ddpg, t, i_episode, Date, path)
        # adapt_speed.store_action(t, a)

        # r = np.array([r])
        # ddpg.store_transition(s, aa, 100*r, s_)
        # if ddpg.pointer > MEMORY_CAPACITY and ddpg.pointer%128==0:
        #     ddpg.learn()
            
        s = s_
        # ep_r += r
        a_last = a
        # if t%500:
        #     wb.save(path+'/'+'flow_record.xls')
    # draw number of car and INDE,draw reward
    # if i_episode == 0:
    #     continue
    # line_fig.Draw_Lines(Date, i_episode, path)
    # sys_per.store_xsl(path)
    # EPSILON = EPSILON*0.8 
    env.Render()

# conv_rate.Figure_ConvergenceRate(path)
np.save(path+'/'+'flow_record.npy', flow_record)
del env