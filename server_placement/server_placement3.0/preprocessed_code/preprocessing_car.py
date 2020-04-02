import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import array as array
import sys
import csv
import gc

gc.disable();
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)



BLOCK_SIZE = 5000
L = 10000
INTERVAL = 20

# 区域限制
max0 = 104.1223
min0 = 104.04211
max1 = 30.70454
min1 = 30.65283


min_time = 1538316145.0

FileName = '../../data/chengdushi_1001_1010.csv'
CHUNK_SIZE = 1000


dict_Car = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}


count = 0
counter = 0
# min_time = 999999999999999999
record_ID = 0

chunker = pd.read_csv( FileName, chunksize = CHUNK_SIZE, usecols=[2], engine = 'python')
# print("read successfully!")
# print(chunker)
num = 1 #记录读取轮数

for chunk in chunker:
    print(num)
    num += 1
    chunk = chunk.values.tolist()
    for i in range(CHUNK_SIZE):
        piece = chunk[i][0]
        piece = piece[1:len(piece)-1]
        list1 = piece.split(',')
        for i in range(len(list1)):
            list1[i] = list1[i].split()
            for j in range(len(list1[i])):
                list1[i][j] = float(list1[i][j])
        if list1[0][2] < min_time:
            continue
        arr = np.array(list1)
        # print(arr)

        # for i in range(len(list1)):
        #     list1[i] = list1[i].split()

        # mtime = float(list1[0][2])
        # # print(mtime)
        # if mtime<min_time:
        #     min_time = mtime
        #     record_ID = count
        #     print('min_time = '+str(min_time))
        #     print('record_ID = '+str(record_ID))
        # if mtime < 1538316145:
        #     counter = counter+1
    # # print(arr[0])
        if arr[:,0].max()>=max0 or arr[:,0].min()<=min0 or arr[:,1].max()>=max1 or arr[:,1].min()<=min1:
            continue

        arr_sorted = []
        ii = int((arr[0,2]-min_time)/INTERVAL)
        for i in range(len(arr)):
            if i == 0:
                arr_sorted.append([L*(arr[i,0]-min0)/(max0-min0), L*(arr[i,1]-min1)/(max1-min1), ii])
            elif int((arr[i,2]-min_time)/INTERVAL) != int((arr[i-1,2]-min_time)/INTERVAL):
                ii = ii+1
                arr_sorted.append([L*(arr[i,0]-min0)/(max0-min0), L*(arr[i,1]-min1)/(max1-min1), ii])
            else:
                pass
        arr_sorted = np.array(arr_sorted)

        if int(arr_sorted[0][2]/4320) <= 9:
            dict_Car[int(arr_sorted[0][2]/4320)].update({count:arr_sorted})
            count = count+1 

        # if count>5:
        #     break   

    if num > 500:
        break

# print(dict_Car)
gc.enable();

print("begin data store!")
for i in range(10):
    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1000+i+1)+'.txt','w')
    f.write(str(dict_Car[i]))
    f.close()