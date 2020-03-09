import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEMORY_CAPACITY = 10000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32


class ANet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # print(x)
        x1 = self.fc1(x)
        x2 = F.relu(x1)
        x3 = self.out(x2)
        x4 = F.relu(x3)
        action_value = x4*2
        return action_value

class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 50)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 50)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50,1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.memory = np.zeros((MEMORY_CAPACITY, 2*s_dim+a_dim+1))
        self.pointer = 0

        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # print(s)
        # print(self.Actor_eval(s)[0])
        return self.Actor_eval(s)[0].detach()

    def learn(self):
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)') 

        indices = np.random.choice(MEMORY_CAPACITY, size = BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, 0: self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim+self.a_dim])
        br = torch.FloatTensor(bt[:, self.s_dim+self.a_dim: self.s_dim+self.a_dim+1])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

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

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1    



