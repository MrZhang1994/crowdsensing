import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import convlstm

MEMORY_CAPACITY = config.MEMORY_CAPACITY
LR_A = config.LR_A    # learning rate for actor
LR_C = config.LR_C    # learning rate for critic
GAMMA = config.GAMMA     # reward discount
TAU = config.TAU      # soft replacement
BATCH_SIZE = config.BATCH_SIZE
RESOLUTION_x = config.RESOLUTION_x
RESOLUTION_y = config.RESOLUTION_y
RESOLUTION_sum = RESOLUTION_x*RESOLUTION_y
# EPSILON = 0.2
KS = int(RESOLUTION_y/10)
use_gpu = config.use_gpu

CONVLSTM_LAYERS = config.CONVLSTM_LAYERS
CLOSEST_IDLE_NUM = config.CLOSEST_IDLE_NUM

# x = torch.rand((32, 10, 1, 1, 1910))
# xx = torch.zeros((32, 10, 1, 1, 2000))
# xx[:,:,:,:,:1910] = x
# y = xx.view(32, 10, 1, 40, 50)
# # x = torch.rand((32, 10, 1, 50, 40))
# convlstm = convlstm.ConvLSTM(input_dim = 1, hidden_dim = 1, kernel_size = (3,3), num_layers = 5, batch_first=True)
# _, last_states = convlstm(y)
# h = last_states[0][0]  # 0 for layer index, 0 for h index
# print(h.shape)

class ANet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.convlstm = convlstm.ConvLSTM(input_dim = 1, hidden_dim = 1, kernel_size = (KS,KS), num_layers = CONVLSTM_LAYERS, batch_first=True)
        self.fc1 = nn.Linear(RESOLUTION_sum, 2*a_dim)
        self.fc1.weight.data.normal_(0, 0.1)

        # self.out = nn.Linear(1000, a_dim)
        # self.out.weight.data.normal_(0, 0.1)        

    def forward(self, s):
        # print(s.shape)

        xx = s.view(1, 10, 1, RESOLUTION_x, RESOLUTION_y)
        # print('xx is')
        # print(xx)
        # print('xx shape is')
        # print(xx.shape)
        _, last_states = self.convlstm(xx)
        h = last_states[0][0]
        # print(h.shape)

        x = h.view(1, 1, 1, RESOLUTION_sum)
        x = x[0][0]
        x = F.leaky_relu(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        # x = self.out(x)
        # x = torch.sigmoid(x)
        action_value = x
        return action_value

class CNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.convlstm = convlstm.ConvLSTM(input_dim = 1, hidden_dim = 1, kernel_size = (KS,KS), num_layers = CONVLSTM_LAYERS, batch_first=True)
        self.fcs = nn.Linear(RESOLUTION_sum, 1000)
        self.fcs.weight.data.normal_(0.1, 0.02)
        self.fca = nn.Linear(2*a_dim, 1000)
        self.fca.weight.data.normal_(0.1, 0.02)
        self.out = nn.Linear(1000,1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # print(s.shape)

        xx = s.view(1, 10, 1, RESOLUTION_x, RESOLUTION_y)
        _, last_states = self.convlstm(xx)
        h = last_states[0][0]
        x = h.view(1, 1, 1, RESOLUTION_sum)
        x = x[0][0]
        x = F.leaky_relu(x)

        x = self.fcs(x)
        y = self.fca(a)
        net = F.leaky_relu(x+y)
        actions_value = self.out(net)
        return actions_value

class PDDPG(object):
    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim

        self.memory = []
        self.pointer = 0

        self.learn_time = 0

        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)

        # self.Actor_eval.load_state_dict(torch.load('result/0512/0512_1015/Actor_eval.pkl'))
        # self.Actor_eval.eval()
        # self.Actor_target.load_state_dict(torch.load('result/0512/0512_1015/Actor_target.pkl'))
        # self.Actor_target.eval()
        # self.Critic_eval.load_state_dict(torch.load('result/0512/0512_1015/Critic_eval.pkl'))
        # self.Critic_eval.eval()
        # self.Critic_target.load_state_dict(torch.load('result/0512/0512_1015/Critic_target.pkl'))
        # self.Critic_target.eval()

        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()
        if use_gpu:
            self.Actor_eval = self.Actor_eval.cuda()
            self.Actor_target = self.Actor_target.cuda()
            self.Critic_eval = self.Critic_eval.cuda()
            self.Critic_target = self.Critic_target.cuda()
            self.loss_td = self.loss_td.cuda()

    def choose_action_0(self, state):
        ss = torch.FloatTensor(state)
        if use_gpu:
            ss = ss.cuda()        
        # print(ss)
        action = self.Actor_eval(ss)
        # print(action)
        action = action.detach()
        # print(action)
        if use_gpu:
            action = action.cpu()
        
        return action

    def choose_action_1(self, state, var):
        ss = torch.FloatTensor(state)
        if use_gpu:
            ss = ss.cuda()  
        action = self.Actor_eval(ss)
        action = action.detach()
        if use_gpu:
            action = action.cpu()

        action = action.numpy()
        action = np.random.normal(action, var)
        action = torch.from_numpy(action)
        
        return action

    def choose_action_2(self, state, s, env, a_last, EPSILON):
        ss = torch.FloatTensor(state)
        if use_gpu:
            ss = ss.cuda()  
        action = self.Actor_eval(ss)
        # print(action)
        action = action.detach()


        if use_gpu:
            action = action.cpu()
        action = action.numpy()
        # print(s)
        # print(action)
        # action = np.expand_dims(action, axis=0)
        # print(action)
        # print(state.shape)
        # print(s.shape)
        for i in range(self.a_dim):

            IDLE_AROUND_FLAG = 0
            for j in range(CLOSEST_IDLE_NUM):
                CLOSEST_CAR_id = int(env.list_INDE_object[i].neighbor[j,0])
                if s[0,CLOSEST_CAR_id] == 0 and a_last[CLOSEST_CAR_id] == 1:
                    IDLE_AROUND_FLAG = IDLE_AROUND_FLAG+1

            if ((s[0,i] == 0 and action[0,i] > 0.5) or (IDLE_AROUND_FLAG == CLOSEST_IDLE_NUM)) and np.random.uniform() <= EPSILON:
            # if s[0,i] == 0 and action[0,i] > 0.5:
                action[0,i] = 0

        action = torch.from_numpy(action)
        
        return action



    def learn(self):
        # if self.learn_time != 0 and self.learn_time%500 == 0:
        #     for p_c in self.ctrain.param_groups:
        #         p_c['lr'] *= 0.9
        #     for p_a in self.atrain.param_groups:
        #         p_a['lr'] *= 0.9

        self.learn_time += 1
        
        # for x in self.Actor_target.state_dict().keys():
        #     eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        # for x in self.Critic_target.state_dict().keys():
        #     eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)') 

        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - TAU) + param.data*TAU)
        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - TAU) + param.data*TAU)
        


        indices = np.random.randint(low = 0, high=MEMORY_CAPACITY)
        bt = self.memory[indices]
        # print('use_gpu = '+str(use_gpu))
        # if use_gpu:
        #     bt = bt.cuda()
        
        bs = torch.FloatTensor(bt[0].astype(np.float32))
        ba = torch.FloatTensor(bt[1].astype(np.float32))
        br = torch.FloatTensor(bt[2].astype(np.float32))
        bs_ = torch.FloatTensor(bt[3].astype(np.float32))

        if use_gpu:
            bs = bs.cuda()
            ba = ba.cuda()
            br = br.cuda()
            bs_ = bs_.cuda()
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

        if use_gpu:
            loss_a = loss_a.cpu()
            td_error = td_error.cpu()
        return float(loss_a), float(td_error)


    def store_transition(self, s, a, r, s_):
        # print(s)
        # print(a)
        # print(r)
        # print(s_)
        # transition = np.hstack((s, a, r, s_))
        if self.pointer < MEMORY_CAPACITY:
            self.memory.append([s, a, r, s_])
        else:  
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index] = [s, a, r, s_]
        self.pointer += 1 

    def save_model(self, path):
        if use_gpu:
            self.Actor_eval = self.Actor_eval.cpu()
            self.Actor_target = self.Actor_target.cpu()
            self.Critic_eval = self.Critic_eval.cpu()
            self.Critic_target = self.Critic_target.cpu()            
        torch.save(self.Actor_eval.state_dict(), path+'/Actor_eval.pkl')
        torch.save(self.Actor_target.state_dict(), path+'/Actor_target.pkl')
        torch.save(self.Critic_eval.state_dict(), path+'/Critic_eval.pkl')
        torch.save(self.Critic_target.state_dict(), path+'/Critic_target.pkl')
        if use_gpu:
            self.Actor_eval = self.Actor_eval.cuda()
            self.Actor_target = self.Actor_target.cuda()
            self.Critic_eval = self.Critic_eval.cuda()
            self.Critic_target = self.Critic_target.cuda() 