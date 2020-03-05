import numpy as np
from numpy import array as array
import random
from random import choice
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# from matplotlib import pyplot as plt
# import pandas as pd

# some parameters
L = 30 # the numbers of rows and columns
SLOTNUM = 200 # the numbers of slots used
MAX_QUEUING_TIME = 1000

# about utilization ratio punishment
CAPACITY = 5 # how many cars one INDE can serve 
U_BEST = 0.7 # as the name say
PHI_1 = 1 # means the punishment of utilization ratio when u<u_best
PHI_2 = 1 #means the punishment of utilization ratio when u>u_best

# about INDE cost
ALPHA = 1 # means the cost of maintaining a basic station INDE
# BETA = 1 # means the cost of maintaining a car INDE

# about centent loss punishment
OMEGA = 1 # means the punishment of centent loss

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000




N_ACTIONS = 1989
N_STATES = 1989
ENV_A_SHAPE = 0   # have not known what it is, just set it 0 temporarily


'''
class Net and DQN are class about DRL,and then can be
changed into DDPG or other DRLs
'''
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Car:
    def __init__(self, arr):
        self.arr = arr
        self.start = int(arr[0,2])
        self.pos = arr[0,0:2]
        self.INDE = -1
        self.distance = 99999
        self.delay = 99999
    def update(self, t, dict_bt_cell, rho, list_bt_object):
        self.pos = self.arr[t-self.start,0:2]
        self.update_INDE(dict_bt_cell, rho, list_bt_object)
        # if self.INDE != -1:
        #     self.delay = abs(self.pos[0]-(list_bt_object[self.INDE]).pos[0])+abs(self.pos[1]-(list_bt_object[self.INDE]).pos[1])
        
        # if self.delay > 500:
        #     '''
        #     find a new INDE
        #     '''
        #     self.update_INDE(dict_bt_cell, list_bt_object)

        # if self.INDE != -1:
        #     self.delay = abs(self.pos[0]-(bt_pos[self.INDE])[0])+abs(self.pos[1]-(bt_pos[self.INDE])[1])                  

        return self.INDE

    def calculate_delay(self, INDE_id, rho, list_bt_object):
        if INDE_id == -1:
            self.delay = 99999
        else:
            if list_bt_object[INDE_id].INDEtype == 0: # a bt INDE
                access_delay = 10
                x = int(600*list_bt_object[INDE_id].pos[0]/10000)
                y = int(600*list_bt_object[INDE_id].pos[1]/10000)
                queuing_delay = MAX_QUEUING_TIME*rho[x][y]
                self.delay = access_delay+queuing_delay
            else: # a car INDE
                self.distance = np.sqrt(np.square(self.pos[0]-list_bt_object[INDE_id].pos[0])+np.square(self.pos[1]-list_bt_object[INDE_id].pos[1]))
                if self.distance <= 400:
                    self.delay = 5
                else:
                    self.delay = 400

    def update_INDE(self, dict_bt_cell, rho, list_bt_object):
        cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
        if dict_bt_cell[cell_number]:
            # self.INDE = choice(dict_bt_cell[cell_number])
            for i in range(len(dict_bt_cell[cell_number])):
                pos0 = list_bt_object[((dict_bt_cell[cell_number])[i])].pos[0]
                pos1 = list_bt_object[((dict_bt_cell[cell_number])[i])].pos[1]
                new_distance = np.sqrt(np.square(self.pos[0]-pos0)+np.square(self.pos[1]-pos1))
                if new_distance < self.distance and list_bt_object[((dict_bt_cell[cell_number])[i])].report_load_rate()<1:
                    self.INDE = (dict_bt_cell[cell_number])[i]
                    self.distance = new_distance         
            # if self.delay >500:
            #     self.INDE = -1
        else:
            self.INDE = -1
        self.calculate_delay(self.INDE, rho, list_bt_object)
        # else:
        #     if cell_number == 0:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #     elif cell_number == 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])                    
        #     elif cell_number == 90:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number == 99:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number < 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #     elif cell_number > 90:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number%10 == 0:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number%10 == 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     else:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])         

class Bt:
    def __init__(self, ID, INDEtype, arr):
        self.id = ID
        self.INDEtype = INDEtype # INDEtype=1 means it's a car; =0 means it's a bt
        self.arr = arr
        if self.INDEtype == 0:
            self.pos = arr
        else:
            self.pos = arr[0,:]
        # self.pos[0] = pos[0]
        # self.pos[1] = pos[1]
        self.state = 0
        self.ConectNum = 0
        self.timestamp = 0
        self.arrlen = len(self.arr)

    def bt_update(self):
        if self.INDEtype == 1: # means it's a carINDE
            if self.timestamp >= self.arrlen:
                self.timestamp = 0
            self.pos = self.arr[self.timestamp,:]
            self.timestamp = self.timestamp+1


    def open(self):
        self.state = 1
        '''
        add itself into open list
        '''
    def close(self):
        self.state = 0
        self.ConectNum = 0
        '''
        delete itself frome open list
        '''
    def connect(self):
        if self.state == 1:
            self.ConectNum = self.ConectNum+1
        else:
            print('the INDE is not open!')

    def disconnect(self):
        if self.ConectNum != 0:
            self.ConectNum = self.ConectNum-1
        else:
            print('no car connect to the INDE now, cannot do the disconnection')

    def report_load_rate(self):
        if self.state == 1:
            load_rate = self.ConectNum/CAPACITY
        else:
            load_rate = U_BEST
        return(load_rate)

class Env:
    '''
    about the envornment
    '''
    def __init__(self):
        '''initializing'''
        self.bt_pos = np.loadtxt('bt_inside_1910.txt')
        self.rho = np.loadtxt('rho.txt')
        f = open('dict_Car.txt','r')
        a = f.read()
        self.dict_Car = eval(a)
        f.close()
        f = open('dict_INDECar.txt','r')
        a = f.read()
        self.dict_IDNECar = eval(a)
        f.close()

        '''
        some list
        '''
        self.list_bt_object = []
        self.dict_existing_car = {}
        self.car_id = 0
        self.dict_open_bt = {}
        self.load_rate = np.zeros((len(self.bt_pos)+len(self.dict_IDNECar),1))
        self.dict_bt_cell = {}
        '''
        create object
        '''
        tempid = 0
        for i in range(len(self.bt_pos)):
            self.list_bt_object.append(Bt(tempid, 0, self.bt_pos[i,:]))
            tempid += 1
        for key in self.dict_IDNECar:
            self.list_bt_object.append(Bt(tempid, 1, self.dict_IDNECar[key]))
            tempid += 1

            # cell_number = 10*int(self.bt_pos[i,0]/(10000/L))+int(self.bt_pos[i,1]/(10000/L))
            # if self.dict_bt_cell[cell_number]:
            #     self.dict_bt_cell[cell_number].append(i)
            # else:
            #     self.dict_bt_cell.update({dict_bt_cell[cell_number]:[i]})
        # for i in range(20):
        #     self.dict_existing_car.update({self.car_id:Car(self.dict_Car[i])})
        #     self.car_id = self.car_id+1

        # figure
        plt.ion()


    def update(self, a, a_last, t):
        '''
        an updation through time
        '''
        # update BTs' openning state through action a
        for i in range(len(self.list_bt_object)):
            self.list_bt_object[i].bt_update()
            if a[i] == 1 and a_last[i] == 0:
                self.list_bt_object[i].open()
                self.dict_open_bt.update({i:[]})
                cell_number = 10*int(self.list_bt_object[i].pos[0]/1000)+int(self.list_bt_object[i].pos[1]/1000)
                if cell_number in self.dict_bt_cell.keys():
                    if i not in self.dict_bt_cell[cell_number]:
                        self.dict_bt_cell[cell_number].append(i)
                else:
                    self.dict_bt_cell.update({cell_number:[i]})  

            elif a[i] == 0 and a_last[i] == 1:
                self.list_bt_object[i].close()
                cell_number = 10*int(self.list_bt_object[i].pos[0]/1000)+int(self.list_bt_object[i].pos[1]/1000)
                self.dict_bt_cell[cell_number].remove(i)

                if self.dict_open_bt[i]:
                    for j in range(len(self.dict_open_bt[i])):
                        # print(self.dict_open_bt[i][j])
                        self.dict_existing_car[self.dict_open_bt[i][j]].INDE = -1
                        self.dict_existing_car[self.dict_open_bt[i][j]].delay = 99999
                del self.dict_open_bt[i]

        # add new car to the environment
        for i in range(20):
            if (i+20*t) in self.dict_Car.keys():
                self.dict_existing_car.update({self.car_id:Car(self.dict_Car[i+20*t])})
                self.car_id = self.car_id+1
            else:
                break

        # cars update their position and maybe find a new INDE

        for key in self.dict_existing_car:
            if self.dict_existing_car[key].INDE != -1:
                # print(self.dict_existing_car[key].INDE)
                self.list_bt_object[self.dict_existing_car[key].INDE].disconnect()
                self.dict_open_bt[self.dict_existing_car[key].INDE].remove(key)
            chosen_INDE = self.dict_existing_car[key].update(t, self.dict_bt_cell, self.rho, self.list_bt_object)
            if chosen_INDE != -1:
                self.list_bt_object[chosen_INDE].connect()
                self.dict_open_bt[chosen_INDE].append(key)

        # calculate load rate and create system log
        '''
        PrintLogType = 0 means print all INDE
        PrintLogType = 1 means print connected INDE
        PrintLogType = 2 means print open INDE
        '''
        PrintLogType = 1
        if t == 0:
            f = open('log'+str(PrintLogType)+'.txt', 'w')
        else:
            f = open('log'+str(PrintLogType)+'.txt', 'a')
        f.write('================================ t = '+str(t)+' =======================================\n'+'\n')
        f.close()
        if PrintLogType == 0:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                f = open('log'+str(PrintLogType)+'.txt', 'a')
                log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                log += 'loadrate='+str(self.load_rate[i])+'||'
                log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                f.write(log)
                if i in self.dict_open_bt.keys(): 
                    if self.dict_open_bt[i]:
                        f.write('connecting cars information:\n')
                        for j in range(len(self.dict_open_bt[i])):
                            log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                            log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                            log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                            log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                            f.write(log)
                f.write('\n')           
                f.close()
        elif PrintLogType == 1:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                if self.list_bt_object[i].ConectNum != 0:
                    f = open('log'+str(PrintLogType)+'.txt', 'a')
                    log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                    log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                    log += 'loadrate='+str(self.load_rate[i])+'||'
                    log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                    log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                    f.write(log)
                    if i in self.dict_open_bt.keys(): 
                        if self.dict_open_bt[i]:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.dict_open_bt[i])):
                                log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                                f.write(log)
                    f.write('\n')           
                    f.close()            
        elif PrintLogType == 2:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                if self.list_bt_object[i].state != 0:
                    f = open('log'+str(PrintLogType)+'.txt', 'a')
                    log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                    log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                    log += 'loadrate='+str(self.load_rate[i])+'||'
                    log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                    log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                    f.write(log)
                    if i in self.dict_open_bt.keys(): 
                        if self.dict_open_bt[i]:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.dict_open_bt[i])):
                                log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                                f.write(log)
                    f.write('\n')          
                    f.close()
        f = open('log'+str(PrintLogType)+'.txt', 'a')
        f.write('Cars that dont have a INDE:'+'\n')
        for key in list(self.dict_existing_car.keys()):
            if self.dict_existing_car[key].INDE == -1:
                log = 'carID='+str(key)+'||'+'pos='+str(self.dict_existing_car[key].pos)+'||'+'delay='+str(self.dict_existing_car[key].delay)+'\n'
                f.write(log)
        f.write('\n')
        f.close()

        # del cars
        for key in list(self.dict_existing_car.keys()):
            # if t == self.dict_existing_car[key].arr[-1,2]:
            if t == self.dict_existing_car[key].arr[-1,2]:
                if self.dict_existing_car[key].INDE != -1:
                    self.list_bt_object[self.dict_existing_car[key].INDE].disconnect()
                    (self.dict_open_bt[self.dict_existing_car[key].INDE]).remove(key)
                del self.dict_existing_car[key]

        # # draw picture
        # cars = []
        # for key in self.dict_existing_car:
        #     cars.append([self.dict_existing_car[key].pos[0],self.dict_existing_car[key].pos[1]])
        # cars = np.array(cars)
        # bts = []
        # for key in self.dict_open_bt:
        #     bts.append([self.list_bt_object[key].pos[0],self.list_bt_object[key].pos[1]])
        # bts = np.array(bts)
        # # print(bts)
        # plt.clf()
        # if len(cars)>0:
        #     plt.scatter(cars[:,0], cars[:,1], c = 'b', marker = '.')
        # if len(bts)>0:
        #     plt.scatter(bts[:,0], bts[:,1], c = 'r', marker = '.')
        # plt.xlim(0, 10000)
        # plt.ylim(0, 10000)
        # plt.show()
        # plt.savefig('result_images/'+str(t)+'.jpg')
        # plt.pause(0.2)
        # # plt.ioff()
        


        return self.load_rate


        
'''
envirment test module
'''
# create random actions which are used to test
OPENRATE = 1
aa = np.zeros((SLOTNUM,N_ACTIONS))
for i in range(SLOTNUM):
    for j in range(N_ACTIONS):
        if random.random() <= OPENRATE:
            aa[i,j] = 1
# aa = np.around(aa)


'''
main program
'''
dqn = DQN()
env = Env()
for i_episode in range(1):
    # a = np.ravel(np.zeros((1,N_ACTIONS)))
    a = np.ravel(np.ones((1,N_ACTIONS)))
    a_last = np.ravel(np.zeros((1,N_ACTIONS)))
    s = np.ravel(np.zeros((1,N_STATES)))
    s_ = np.ravel(np.zeros((1,N_STATES)))
    s = env.update(a,a_last,0)


    # s[0:N_ACTIONS] = a
    # s[N_ACTIONS:N_STATES] = np.ravel(utilization_ratio)

    s_ = s
    a_last = a
    ep_r = 0
    for t in range(1,SLOTNUM):
        # a = dqn.choose_action(s)
        a = aa[t,:]
        # print(a)
        print(t)
        s_ = env.update(a, a_last, t)
        # s_[0:N_ACTIONS] = a
        # s_[N_ACTIONS:N_STATES] = np.ravel(utilization_ratio)

        # calculate INDE cost
        cost_open = ALPHA*np.sum(a)

        # calculate utilization ratio punishment
        puni_utilization = 0
        for i in range(N_STATES):
            if s_[i]<=U_BEST:
                puni_utilization += PHI_1*(U_BEST-s_[i])
            else:
                puni_utilization += PHI_2*(s_[i]-U_BEST)

        # calculate centent loss punishment
        puni_contentloss = 0
        for i in range(N_ACTIONS):
            if a[i] == 1 and a[i] == 0:
                puni_contentloss += OMEGA

        # calculate total r
        r = -cost_open-puni_utilization-puni_contentloss

        # dqn.store_transition(s, a, r, s_)

        ep_r += r
        # if dqn.memory_counter > MEMORY_CAPACITY:
        #     dqn.learn()
            
        s = s_
        a_last = a
