import numpy as np
from numpy import array as array
import torch
import torch.nn as nn
import torch.nn.functional as F

# from matplotlib import pyplot as plt
# import pandas as pd

# some parameters
L = 30 # the numbers of rows and columns
SLOTNUM = 300 # the numbers of slots used

# about utilization ratio punishment
CAPACITY = 2 # how many cars one INDE can serve 
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



bt_count = np.loadtxt('bt_count.txt')
N_ACTIONS = int(np.sum(bt_count))
N_STATES = int(np.sum(bt_count)+L*L)
ENV_A_SHAPE = 0   # have not known what it is, just set it 0 temporarily


# read processed car data
f = open('dict_Car_3298_processed_2.txt','r')
a = f.read()
dict_Car = eval(a)

# count number of cars every cell every slot 
car_count = np.zeros((L,L,SLOTNUM))
for key in dict_Car:
    arr_sorted = dict_Car[key]
    for i in range(len(arr_sorted)):
        if arr_sorted[i,2]<SLOTNUM:
            car_count[int(arr_sorted[i,0]),int(arr_sorted[i,1]),int(arr_sorted[i,2])] += 1
        else:
            break
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

class Env:
    '''
    about the envornment
    '''
    def __init__(self, bt_count, car_count):
        '''initializing'''
        self.open_count = np.ones((L,L))
        self.bt_count = bt_count
        self.utilization_ratio = np.zeros((L,L))
        self.car_count = car_count
    def update(self, *a, t):
        temp = 0
        for i in range(L):
            for j in range(L):
                self.open_count[i,j] = np.sum(a[temp:int(temp+self.bt_count[i,j])])
                temp = int(temp+self.bt_count[i,j])
                if self.open_count[i,j] == 0 and self.car_count[i,j,t] == 0:
                    self.utilization_ratio[i,j] = U_BEST
                elif self.open_count[i,j] == 0 and self.car_count[i,j,t] != 0:
                    self.utilization_ratio[i,j] = 100 # I gives a very big punishment to this situation.Need to be considered
                self.open_count[i,j] = np.sum(a[temp:temp+self.bt_count[i,j]])
                if self.open_count[i,j] == 0 and self.car_count[i,j,t] == 0:
                    self.utilization_ratio[i,j] = U_BEST
                elif self.open_count[i,j] == 0 and self.car_count[i,j,t] != 0:
                    self.utilization_ration[i,j] = 100 # I gives a very big punishment to this situation.Need to be considered
                else:
                    self.utilization_ratio[i,j] = self.car_count[i,j,t]/(CAPACITY*self.open_count[i,j])
        return self.utilization_ratio

        
dqn = DQN()
env = Env(bt_count, car_count)

for i_episode in range(400):
    a = np.ravel(np.zeros((1,N_ACTIONS)))
    utilization_ratio = env.update(a, t = 0)
    s = np.ravel(np.zeros((1,N_STATES)))
    s[0:N_ACTIONS] = a
    s[N_ACTIONS:N_STATES] = np.ravel(utilization_ratio)
    s_ = s
    ep_r = 0
    for t in range(1,SLOTNUM):
        a = dqn.choose_action(s)
        utilization_ratio = env.update(a, t = t)
        t = t+1
        s_[0:N_ACTIONS] = a
        s_[N_ACTIONS:N_STATES] = np.ravel(utilization_ratio)
        cost_open = ALPHA*np.sum(a)
        puni_utilization = 0
        for i in range(L):
            for j in range(L):
                if utilization_ratio[i,j]<=U_BEST:
                    puni_utilization += PHI_1*(U_BEST-utilization_ratio[i,j])
                else:
                    puni_utilization += PHI_2*(utilization_ratio[i,j]-U_BEST)
        puni_contentloss = 0
        for i in range(N_ACTIONS):
            if s[i] == 1 and s_[i] == 0:
                puni_contentloss += OMEGA
        r = -cost_open-puni_utilization-puni_contentloss

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            
        s = s_