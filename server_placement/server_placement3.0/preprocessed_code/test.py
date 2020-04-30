import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# import pandas as pd
L = 10000
SAMPLE = 5
max0 = 104.1223
min0 = 104.04211
max1 = 30.70454
min1 = 30.65283

bt = np.loadtxt('../../data/chengdu_bt.txt')

# plt.figure(1)
# plt.scatter(bt[:,0], bt[:, 1], marker = '.')
# plt.show()

print(bt.shape)
count = 0
for i in range(len(bt)):
    if bt[i][0]>=min0 and bt[i][0]<=max0 and bt[i][1]>=min1 and bt[i][1]<=max1 and i%SAMPLE==0:
        count = count+1;
print(count)
bt_inside = np.zeros((count,2))
count = 0
temp = 0
for i in range(len(bt)):
    if bt[i][0]>=min0 and bt[i][0]<=max0 and bt[i][1]>=min1 and bt[i][1]<=max1 and i%SAMPLE==0:
        bt_inside[count][0] = L*(bt[i][0]-min0)/(max0-min0)
        bt_inside[count][1] = L*(bt[i][1]-min1)/(max1-min1)
        count = count+1;

Grid_len = 20

heat = np.zeros((Grid_len,Grid_len))
for i in range(len(bt_inside)):
    x = int(bt_inside[i][1]*Grid_len/10000)
    y = int(bt_inside[i][0]*Grid_len/10000)
    heat[x,y] += 1

heatmap = sns.heatmap(heat)
plt.show()

plt.figure(1)
plt.scatter(bt_inside[:,0], bt_inside[:,1], marker = '.')
plt.show()

# del bt
# print(count)
# np.savetxt('../preprocessed_data/bt_inside_'+str(count)+'.txt',bt_inside)
