import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd
L = 30

bt = np.loadtxt('chengdu_bt.txt')
print(bt.shape)
count = 0
for i in range(len(bt)):
    if bt[i][0]>=104.028739 and bt[i][0]<=104.122597 and bt[i][1]>=30.615096 and bt[i][1]<=30.70455:
        count = count+1;
print(count)
bt_inside = np.zeros((count,2))
count = 0
for i in range(len(bt)):
    if bt[i][0]>=104.028739 and bt[i][0]<=104.122597 and bt[i][1]>=30.615096 and bt[i][1]<=30.70455:
        bt_inside[count][0] = L*(bt[i][0]-104.028739)/(104.122597-104.028739)
        bt_inside[count][1] = L*(bt[i][1]-30.615096)/(30.70455-30.615096)
        count = count+1;
del bt
print(bt_inside.shape)
bt_count = np.ones((L,L))
for i in range(len(bt_inside)):
    if bt_count[int(bt_inside[i][0])][int(bt_inside[i][1])]<=9:
        bt_count[int(bt_inside[i][0])][int(bt_inside[i][1])] += 1

print(bt_count)
np.savetxt('bt_count.txt',bt_count)
