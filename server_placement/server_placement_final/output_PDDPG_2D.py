# Libs used
import numpy as np
from matplotlib import pyplot as plt
import copy
import gc
import math
import xlwt


# Modules used
import config
import envirment_PDDPG

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
L1_INDE_CAPACITY = config.L1_INDE_CAPACITY

def Calculate_disconnect_outofdelay(env):
    disconnect_rate = 1-(env.Print_System_Performance())[1]
    L1_Delay = env.Get_L1_Car_Delay()
    L1_Delay_sum = 0
    L1_Delay_count = 0
    L1_Delay_outrage_count = 0
    Avg_Delay = 0
    Delay_Outage_Rate = 0
    for i in range(a_dim):
        if L1_Delay[i] != 0:
            L1_Delay_sum = L1_Delay_sum+L1_Delay[i]
            L1_Delay_count = L1_Delay_count+1
        if L1_Delay[i]>50:
            L1_Delay_outrage_count = L1_Delay_outrage_count+1
    if L1_Delay_count != 0:
        Avg_Delay = L1_Delay_sum/L1_Delay_count
        Delay_Outage_Rate = L1_Delay_outrage_count/L1_Delay_count

    return disconnect_rate, Avg_Delay, Delay_Outage_Rate

def Calculate_Reward(a, uti, disconnect_rate, Delay_Outage_Rate):
    # calculate INDE cost
    # cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])
    cost_open = np.sum(a[0:Bt_INDE_num])

    # calculate utilization ratio reward
    reward_utilization = 0
    for i in range(a_dim):
        # if uti[i]<=0.6:
        #     reward_utilization += (1/0.6)*uti[i]
        # elif uti[i]<=1:
        #     reward_utilization += 1
        if uti[i]<=1:
            reward_utilization += uti[i]
        else:
            reward_utilization += (-uti[i]*uti[i]+uti[i]+1)
    # calculate total r
    # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
    r = 6*reward_utilization*(L1_INDE_CAPACITY/5)-3*cost_open-1000*disconnect_rate-1000*Delay_Outage_Rate
    return r

def Calculate_Reward_new(a, uti, disconnect_rate, Delay_Outage_Rate, env):
    # calculate INDE cost
    # cost_open = ALPHA*np.sum(a[0:Bt_INDE_num])+BETA*np.sum(a[Bt_INDE_num:])
    cost_open = np.sum(a[0:Bt_INDE_num])

    # calculate utilization ratio reward
    reward_utilization = 0
    for i in range(a_dim):
        # if uti[i]<=0.6:
        #     reward_utilization += (1/0.6)*uti[i]
        # elif uti[i]<=1:
        #     reward_utilization += 1
        if uti[i]<=1:
            reward_utilization += uti[i]
        else:
            reward_utilization += (-uti[i]*uti[i]+uti[i]+1)
    # calculate total r
    # r = 1000*connect_rate-cost_open-puni_utilization-puni_contentloss
    r = 6*reward_utilization*(L1_INDE_CAPACITY/5)-3*cost_open-5*disconnect_rate*(env.Print_System_Performance())[0]
    return r

class SystemPerformance:
    def __init__(self, a_dim, ):
        self.a_dim = a_dim
        # self.spf = None
        self.wb = xlwt.Workbook(encoding="utf-8")
        self.ws = None


    def reset(self, Date, i_episode, State):
        if State == 0:
            self.ws = self.wb.add_sheet('Train_'+str(Date)+'_'+str(i_episode), cell_overwrite_ok=True)
        elif State == 1:
            self.ws = self.wb.add_sheet('Test_'+str(Date), cell_overwrite_ok=True)
        title = ['Time', 'Car Num', 'Open INDE Num', 'Reward', 'Car Connect Rate', 'Mean Workload', 'Mean Latency', 'Latency Outage Rate', 'AdaptSpeed']
        for i in range(len(title)):
            self.ws.write(0, i, title[i])

    def update(self, env, a, r, disconnect_rate, Avg_Delay, Delay_Outage_Rate, t, path):
        
        car_num = (env.Print_System_Performance())[0]

        connect_rate = 1-disconnect_rate

        uti = np.squeeze(env.load_rate)
        open_uti_rate_sum = 0
        ave_open_uti_rate = 0
        for i in range(self.a_dim):
            if a[i] == 1:
                open_uti_rate_sum += uti[i]
        if np.sum(a)!= 0:
            ave_open_uti_rate = open_uti_rate_sum/np.sum(a)

        self.ws.write(t+1, 0, t)
        self.ws.write(t+1, 1, int(car_num))
        self.ws.write(t+1, 2, int(np.sum(a)))
        self.ws.write(t+1, 3, int(r))
        self.ws.write(t+1, 4, float(connect_rate))
        self.ws.write(t+1, 5, float(ave_open_uti_rate))
        self.ws.write(t+1, 6, float(Avg_Delay))
        self.ws.write(t+1, 7, float(Delay_Outage_Rate))
        # if t%4319 == 0:
        #     self.store_xsl(path)

    def update_AdaptSpeed(self, reward_list, t):
        self.ws.write(t+1, 8, str(reward_list))

    def store_xsl(self, path):
        self.wb.save(path+'/'+'SystemPerformance/SystemPerformance.xls')

class LineFigures:

    def __init__(self, a_dim):
        plt.ion()
        self.a_dim = a_dim
        self.L1_Car_Num_line = np.zeros((SLOTNUM,2))
        self.Reward_line = np.zeros((SLOTNUM,))
        self.Avg_Delay_line = np.zeros((SLOTNUM,))

    def reset(self):
        self.L1_Car_Num_line = np.zeros((SLOTNUM,2))
        self.Reward_line = np.zeros((SLOTNUM,))
        self.Avg_Delay_line = np.zeros((SLOTNUM,))

    def update(self, env, r, t):
        L1_Delay = env.Get_L1_Car_Delay()
        L1_Delay_sum = 0
        L1_Delay_count = 0
        for i in range(self.a_dim):
            if L1_Delay[i] != 0:
                L1_Delay_sum = L1_Delay_sum+L1_Delay[i]
                L1_Delay_count = L1_Delay_count+1
        if L1_Delay_count != 0:
            self.Avg_Delay_line[t] = L1_Delay_sum/L1_Delay_count
        self.L1_Car_Num_line[t,:] = env.Get_L1_Car_Num()        
        self.Reward_line[t] = r

    def Draw_Lines(self, Date, i_episode, path):
        # Draw time as transverse axis
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(range(1,SLOTNUM), self.L1_Car_Num_line[1:,0], color='blue', label='Car number')
        plt.plot(range(1,SLOTNUM), self.L1_Car_Num_line[1:,0]/config.L1_INDE_CAPACITY, color='yellow', label='ideal INDE number')
        plt.plot(range(1,SLOTNUM), self.L1_Car_Num_line[1:,1], color='red', label='INDE number')
        plt.subplot(2,1,2)
        plt.plot(range(1,SLOTNUM), self.Reward_line[1:], color='green', label='Reward')
        # plt.show()
        plt.savefig(path+'/'+'car_L1_reward_figure/i_episode_'+str(Date)+'_'+str(i_episode)+'.png')
        # plt.pause(1)
        # Draw car number as transverse axis
        dict_car_num = {}
        for i in range(1, SLOTNUM):
            if self.L1_Car_Num_line[i,0] in dict_car_num.keys():
                (dict_car_num[self.L1_Car_Num_line[i,0]])[0].append(self.Reward_line[i]) # Reward
                (dict_car_num[self.L1_Car_Num_line[i,0]])[1].append(self.L1_Car_Num_line[i,1]) # INDE number
                (dict_car_num[self.L1_Car_Num_line[i,0]])[2].append(self.Avg_Delay_line[i])
            else:
                dict_car_num.update({self.L1_Car_Num_line[i,0]:[[self.Reward_line[i]], [self.L1_Car_Num_line[i,1]], [self.Avg_Delay_line[i]]]})

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
        plt.savefig(path+'/'+'car_L1_reward_figure/INDE-Car_i_episode_'+str(Date)+'_'+str(i_episode)+'.png')        



class AdaptSpeed:
    def __init__(self, SLOTNUM, a_dim):
        self.action_table = np.zeros((SLOTNUM, a_dim))
        self.a_dim = a_dim
    def reset(self, SLOTNUM, a_dim):
        self.action_table = np.zeros((SLOTNUM, a_dim))

    def store_action(self, t, action):
        self.action_table[t,:] = action

    def Calculate_Reward_in_Other_Action(self, env, other_action, a_last, i_episode, t, Date, SLOTNUM):
        temp_env = copy.deepcopy(env)
        loadrate, mixed_loadrate = temp_env.update_only(other_action, a_last, i_episode, t+(Date-1001)*SLOTNUM)
        temp_s_ = mixed_loadrate
        disconnect_rate, Avg_Delay, Delay_Outage_Rate = output.Calculate_disconnect_outofdelay(env)
        temp_r = Calculate_Reward(other_action, loadrate, disconnect_rate, Delay_Outage_Rate)
        del temp_env
        gc.collect()
        return temp_r,disconnect_rate,Delay_Outage_Rate

    def Calculate_AdaptSpeed(self, env, a_last, i_episode, t, Date, SLOTNUM):
        gap = 5
        reward_list = []
        disconnect_rate_list = []
        Delay_Outage_Rate_list = []
        for i in range(1,10):
            if (t-i*gap) in range(t):
                historical_action = self.action_table[t-i*gap, :]
                reward,disconnect_rate,Delay_Outage_Rate = self.Calculate_Reward_in_Other_Action(env, historical_action, a_last, i_episode, t, Date, SLOTNUM)
                reward_list.append(float(reward))
                disconnect_rate_list.append(float(disconnect_rate))
                Delay_Outage_Rate_list.append(float(Delay_Outage_Rate))
        return [reward_list[::-1], disconnect_rate_list[::-1], Delay_Outage_Rate_list[::-1]]

class ConvergenceRate:
    def __init__(self, MAX_EPISODES, SLOTNUM, a_dim, s_dim):
        self.SLOTNUM = SLOTNUM
        self.CAPACITY = MAX_EPISODES
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.reward_t = np.zeros((SLOTNUM,))
        self.reward_episode = np.zeros((self.CAPACITY,))
        self.ConvergenceRate_list = np.zeros((self.CAPACITY,))
        self.env = None
        self.a = np.zeros((1,a_dim))
        for i in range(a_dim):
            if np.random.uniform()<0.2:
                self.a[0,i] = 1
        self.a = np.ravel(self.a)

        self.a_last = np.ravel(np.zeros((1,a_dim)))
        self.s = np.ravel(np.zeros((1,s_dim)))
        self.s_ = np.ravel(np.zeros((1,s_dim)))
        self.s_ = self.s
        self.a_last = self.a




    def reset(self, Date, path):
        self.env = envirment.Env(Date, path)

        self.a = np.zeros((1,self.a_dim))
        for i in range(self.a_dim):
            if np.random.uniform()<0.2:
                self.a[0,i] = 1
        self.a = np.ravel(self.a)

        self.a_last = np.ravel(np.zeros((1,self.a_dim)))
        self.s = np.ravel(np.zeros((1,self.s_dim)))
        self.s_ = np.ravel(np.zeros((1,self.s_dim)))

        loadrate, mixed_loadrate = self.env.update_only(self.a, self.a_last, 0, 0)
        self.s = mixed_loadrate        

        self.s_ = self.s
        self.a_last = self.a
        self.s = np.transpose(self.s)
        self.s_ = np.transpose(self.s_)

    def store_reward(self, DRL, t, i_episode, Date, path):
        aa = DRL.choose_action_0(self.s)
        self.a = aa[0,:].numpy()

        loadrate, mixed_loadrate = self.env.update_only(self.a, self.a_last, i_episode, t+(Date-1001)*self.SLOTNUM)
        self.s_ = mixed_loadrate
        disconnect_rate, Avg_Delay, Delay_Outage_Rate = Calculate_disconnect_outofdelay(self.env)
        r = output.Calculate_Reward(self.a, loadrate, disconnect_rate, Delay_Outage_Rate)

        self.reward_t[t] = r
        self.s_ = np.transpose(self.s_)
        self.s = self.s_
        self.a_last = self.a
        if t == self.SLOTNUM-1:
            self.reward_episode[i_episode] =  np.mean(self.reward_t, axis = 0)
            self.ConvergenceRate_list[i_episode-1] = self.Calculate_ConvergenceRate(i_episode-1)
            np.save(path+'/'+'ConvergenceRate/ConvergenceRate.npy',self.ConvergenceRate_list)

    def Calculate_ConvergenceRate(self, i):
        if i-2>=0 and i<self.CAPACITY:
            if self.reward_episode[i-1]!=self.reward_episode[i-2] and self.reward_episode[i]!=self.reward_episode[i-1] and self.reward_episode[i+1]!=self.reward_episode[i]:
                return (math.log(abs((self.reward_episode[i+1]-self.reward_episode[i])/(self.reward_episode[i]-self.reward_episode[i-1]))))/(math.log(abs((self.reward_episode[i]-self.reward_episode[i-1])/(self.reward_episode[i-1]-self.reward_episode[i-2]))))
            else:
                return 0
        else:
            return 0

    def Figure_ConvergenceRate(self, path):
        plt.clf()
        plt.plot(range(1, len(self.ConvergenceRate_list)+1), self.ConvergenceRate_list, color='blue')
        plt.savefig(path+'/'+'ConvergenceRate/ConvergenceRate.png')    