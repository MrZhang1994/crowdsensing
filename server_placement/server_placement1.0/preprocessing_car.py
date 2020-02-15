import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

BLOCK_SIZE = 5000
L = 30
INTERVAL = 20


# when BLOCK_SIZE = 5000,min_time is 1541172676.0
min_time = 1541172676.0


with open('chengdushi_1101_1110.csv', 'r') as fin:
    block = []
    for line in fin:
        block.append(line)
        if len(block) > BLOCK_SIZE:
            break
        # print (block[0])
fin.close()

dict_Car = {}
count = 0
# max_len = np.zeros((BLOCK_SIZE,1))
# max_lentime = 0

for k in range(BLOCK_SIZE):
    list1 = eval(block[k][66:])
    list2 = list1[1:len(list1)-1].split(',')
    for i in range(len(list2)):
        list2[i] = list2[i].split()
        for j in range(len(list2[i])):
            list2[i][j] = float(list2[i][j])
    arr = np.array(list2)
    if arr[:,0].max()>=104.122597 or arr[:,0].min()<=104.028739 or arr[:,1].max()>=30.70455 or arr[:,1].min()<=30.615096:
        continue
    valid_len = 1
    for i in range(len(arr)):
        arr[i,0] = int(L*(arr[i,0]-104.028739)/(104.122597-104.028739))
        arr[i,1] = int(L*(arr[i,1]-30.615096)/(30.70455-30.615096))
        arr[i,2] = int((arr[i,2]-min_time)/INTERVAL)
        if i>0 and arr[i,2] != arr[i-1,2]:
            valid_len = valid_len+1
    arr_sorted = np.zeros((valid_len,3))
    ii = 0
    for i in range(len(arr)):
        if i==0:
            arr_sorted[ii,:] = arr[i,:]
            ii = ii+1
        if i>0 and arr[i,2] != arr[i-1,2]:
            arr_sorted[ii,:] = arr[i,:]
            ii = ii+1
    # max_len[k,0] = len(arr_sorted)
    if len(arr_sorted) != 0:
        start_time = arr_sorted[0,2]
        for i in range(len(arr_sorted)):
            arr_sorted[i,2] = arr_sorted[i,2]-start_time+int(count/20)
            # if arr_sorted[i,2]>max_lentime:
            #     max_lentime = arr_sorted[i,2]
        dict_Car.update({count:arr_sorted})
        count = count+1
    # if arr[:,2].max()>max_time:
    #     max_time = arr[:,2].max()
    # if arr[:,2].min()<min_time:
    #     min_time = arr[:,2].min()

# print(np.mean(max_len))
# print(max_lentime)
del list1,list2,arr,arr_sorted


print("begin data store!")
f = open('dict_Car_'+str(count)+'_processed_2.txt','w')
f.write(str(dict_Car))
f.close()