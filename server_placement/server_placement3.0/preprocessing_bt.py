import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd
L = 10000
SAMPLE = 5
max0 = 104.1223
min0 = 104.04211
max1 = 30.70454
min1 = 30.65283

bt = np.loadtxt('../data/chengdu_bt.txt')
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
del bt
print(count)
np.savetxt('bt_inside_'+str(count)+'.txt',bt_inside)
