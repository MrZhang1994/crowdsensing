import numpy as np
from matplotlib import pyplot as plt
from numpy import array as array
import math
import random

LL = 600
# x_iter = (max0-min0)/LL
# y_iter = (max1-min1)/LL

# bt_pos = np.loadtxt('bt_inside_1910.txt')
# bt_count = np.zeros((LL,LL))
# for i in range(len(bt_pos)):
#     x = int(LL*bt_pos[i,0]/10000)
#     y = int(LL*bt_pos[i,1]/10000)
#     bt_count[x,y] += 1
# # print(bt_count)
# print(np.max(bt_count))
# # np.savetxt('bt_count.txt',bt_count)



w_max = 0.012
L = 10
miu = 18
sigma = 2.2

rho = np.zeros((LL,LL))

for i in range(LL):
    for j in range(LL):
        for l in range(L):
            i_l = random.uniform(0,w_max)
            j_l = random.uniform(0,w_max)
            phi_l = random.uniform(0,2*3.1415926)
            psi_l = random.uniform(0,2*3.1415926)
            x = i*10+5
            y = j*10+5
            rho[i][j] += math.cos(i_l*x+phi_l)*math.cos(j_l*y+psi_l)
        rho[i][j] = rho[i][j]*2/np.sqrt(L)
        rho[i][j] = math.exp(sigma*rho[i][j]+miu)

rho_max = np.max(rho)
print(rho_max)
print(np.mean(np.mean(rho)))

for i in range(LL):
    for j in range(LL):
        rho[i][j] = rho[i][j]/rho_max

# print(rho_max)
# print(np.mean(np.mean(rho)))
print(rho)
print(np.mean(np.mean(rho)))
np.savetxt('../preprocessed_data/rho.txt',rho)
