import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

MEMORY_CAPACITY = config.MEMORY_CAPACITY
LR_A = config.LR_A    # learning rate for actor
LR_C = config.LR_C    # learning rate for critic
GAMMA = config.GAMMA     # reward discount
TAU = config.TAU      # soft replacement
BATCH_SIZE = config.BATCH_SIZE
# EPSILON = 0.2

use_gpu = torch.cuda.is_available()

class ANet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        self.fc1 = nn.Linear(s_dim-5+1, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        # self.rnn = nn.LSTM(
        #     input_size = 100,
        #     hidden_size = 100,
        #     num_layers = 1,
        #     batch_first = True
        #     )
        self.out = nn.Linear(100, a_dim)
        self.out.weight.data.normal_(0, 0.1)        

    def forward(self, s):

        x = s.permute(1,0,2)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = x.permute(1,0,2)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.out(x)
        x = torch.sigmoid(x)
        action_value = x
        return action_value

class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        self.fcs = nn.Linear(s_dim-5+1, 100)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 100)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100,1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = s.permute(1,0,2)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = x.permute(1,0,2)
        x = self.fcs(x)
        y = self.fca(a)
        net = F.leaky_relu(x+y)
        actions_value = self.out(net)
        return actions_value

class PDDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.memory = np.zeros((MEMORY_CAPACITY, 2*s_dim+a_dim+1))
        self.pointer = 0
        self.learn_time = 0

        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()
        if use_gpu:
            self.Actor_eval = self.Actor_eval.cuda()
            self.Actor_target = self.Actor_target.cuda()
            self.Critic_eval = self.Critic_eval.cuda()
            self.Critic_target = self.Critic_target.cuda()
            self.loss_td = self.loss_td.cuda()

    def choose_action_0(self, s):
        if use_gpu:
            s = s.cuda()        
        ss = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.Actor_eval(ss)[0].detach()
        if use_gpu:
            action = action.cpu()
        
        return action

    def choose_action_1(self, s, var):
        if use_gpu:
            s = s.cuda()        
        ss = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.Actor_eval(ss)[0].detach()
        if use_gpu:
            action = action.cpu()

        action = action.numpy()
        action = np.random.normal(action, var)
        action = torch.from_numpy(action)
        
        return action

    def choose_action_2(self, s, EPSILON):
        if use_gpu:
            s = s.cuda()        
        ss = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.Actor_eval(ss)[0].detach()
        if use_gpu:
            action = action.cpu()
        action = action.numpy()
        # print(s)
        # print(action)
        # action = np.expand_dims(action, axis=0)
        # print(action)
        for i in range(self.a_dim):

            if s[0,i] == 0 and action[0,i] > 0.5 and np.random.uniform() <= EPSILON:
            # if s[0,i] == 0 and action[0,i] > 0.5:
                action[0,i] = 0
        # if np.sum(a_last) == 0:
        #     for i in range(self.a_dim):
        #         if random.random()<0.2:
        #             action[0,i] = 1
        #         else:
        #             action[0,i] = 0
        # else:    
        #     for i in range(self.a_dim):

        #         if s[0,i] == 0 and action[0,i] > 0.5 and np.random.uniform() <= EPSILON:
        #         # if s[0,i] == 0 and action[0,i] > 0.5:
        #             action[0,i] = 0
        # for i in range(self.a_dim):
        #     if s[0,i] == 1 and i>=1 and np.random.uniform() <= EPSILON:
        #     # if s[0,i] == 0 and action[0,i] > 0.5:
        #         action[0,i-1] = 1 
        action = torch.from_numpy(action)
        return action

    def choose_action_3(self, s, EPSILON):
        if use_gpu:
            s = s.cuda()        
        ss = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = self.Actor_eval(ss)[0].detach()
        if use_gpu:
            action = action.cpu()
        action = action.numpy()
        # print(s)
        # print(action)
        action = np.expand_dims(action, axis=0)
        # print(action)
        for i in range(self.a_dim):
            # if s[0,i] <= 0.4 and action[0,i] == 1 and np.random.uniform() <= EPSILON:
            if s[0,i] == 0 and np.random.uniform()<EPSILON and action[0,i] > 0.5:
                action[0,i] = 0
        for i in range(self.a_dim):
            if s[0,i] == 1:
                action[0,i] = 1
                if i <= self.a_dim-2 and i>=1 and np.random.uniform()<0.3:
                    action[0,i+1] = 1     
                    action[0,i-1] = 1
        action = torch.from_numpy(action)
        return action

    def learn(self):
        # if self.learn_time != 0 and self.learn_time%100 == 0:
        #     for p_c in self.ctrain.param_groups:
        #         p_c['lr'] *= 0.9
        #     for p_a in self.atrain.param_groups:
        #         p_a['lr'] *= 0.9

        self.learn_time += 1
        
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)') 
        
        for i in range(BATCH_SIZE):
            indices = np.random.choice(MEMORY_CAPACITY, size = 1, replace=False, p=None)
            bt = self.memory[indices, :]
            if use_gpu:
                bt = bt.cuda()
            bs = torch.FloatTensor(bt[:, 0: self.s_dim]).unsqueeze(0)
            ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim+self.a_dim])
            br = torch.FloatTensor(bt[:, self.s_dim+self.a_dim: self.s_dim+self.a_dim+1])
            bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).unsqueeze(0)

            # print(bs)
            # print(list(bs.size()))
            a = self.Actor_eval(bs)
            loss_a = self.Critic_eval(bs, a)

            # loss_a = -torch.mean(q)
            self.atrain.zero_grad()
            loss_a.backward()
            self.atrain.step()

            a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q_target = br+GAMMA*q_  # q_target = 负的
            q_v = self.Critic_eval(bs, ba)
            td_error = self.loss_td(q_target, q_v)
            self.ctrain.zero_grad()
            td_error.backward()
            self.ctrain.step()

        return float(loss_a), float(td_error)

    def learn1(self):
        # if self.learn_time != 0 and self.learn_time%1000 == 0:
        #     for p_c in self.ctrain.param_groups:
        #         p_c['lr'] *= 0.9
        #     for p_a in self.atrain.param_groups:
        #         p_a['lr'] *= 0.9

        self.learn_time += 1
        
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)') 
        
        indices = np.random.choice(MEMORY_CAPACITY, size = BATCH_SIZE, replace=False, p=None)
        bt = self.memory[indices, :]
        if use_gpu:
            bt = bt.cuda()
        bs = torch.FloatTensor(bt[:, 0: self.s_dim]).unsqueeze(0)
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim+self.a_dim])
        br = torch.FloatTensor(bt[:, self.s_dim+self.a_dim: self.s_dim+self.a_dim+1])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).unsqueeze(0)

        # print(bs)
        # print(list(bs.size()))
        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)

        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        q_v = self.Critic_eval(bs, ba)
        td_error = self.loss_td(q_target, q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        return float(loss_a), float(td_error)



    def store_transition(self, s, a, r, s_):
        # print(s)
        # print(a)
        # print(r)
        # print(s_)
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1 

    def save_model(self, path):
        torch.save(self.Actor_eval.state_dict(), path+'/Actor_eval.pkl')
        torch.save(self.Actor_target.state_dict(), path+'/Actor_target.pkl')
        torch.save(self.Critic_eval.state_dict(), path+'/Critic_eval.pkl')
        torch.save(self.Critic_target.state_dict(), path+'/Critic_target.pkl')