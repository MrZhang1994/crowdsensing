import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import array as array
import sys
import csv
import gc

#这一部分是提高程序可用的内存上限，防止程序直接卡死
###############################################
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
###############################################
L = 10000 #将经纬度归一化到L*L的区域
INTERVAL = 20   #以多少秒为一个时隙

# 区域限制
max0 = 104.1223     #最东
min0 = 104.04211    #最西
max1 = 30.70454     #最北
min1 = 30.65283     #最南


min_time = 1538336145.0     #最小时间，也就是我们认为1001的0时所在的时间
min_date = 1001             #最小日期

FileName = '../../data/chengdushi_1001_1110.csv'    #要处理的数据集
StartDate = 1001                                    #数据集开始时间
Length = 10                                         #数据集持续的天数


#初始化一个字典，键是日期，值是一个装当天车辆信息的字典
dict_Car = {}
for i in range(StartDate,StartDate+Length):
    dict_Car.update({i:{}})

count = 0   #一个给车辆编号的计数器

CHUNK_SIZE = 1000   #批处理的批大小
chunker = pd.read_csv( FileName, chunksize = CHUNK_SIZE, usecols=[2], engine = 'python')    #分批器

num = 1 #读取的批数的计数器

for chunk in chunker:   #对分批器中的每一批
    print(num)          #打印这是第几批
    num += 1            #批数加1
    chunk = chunk.values.tolist()   #把这一批都转成list
    for i in range(CHUNK_SIZE):     #因为一批里面还是有CHUNK_SIZE条车辆轨迹，要把它们拿出来一一处理
        piece = chunk[i][0]         #拿出一条轨迹
        piece = piece[1:len(piece)-1]   #取出有用部分
        list1 = piece.split(',')        #这时候list里面还是str格式的一整块，所以要通过逗号先分割成很多str数据小块
        for i in range(len(list1)):
            list1[i] = list1[i].split()
            for j in range(len(list1[i])):
                list1[i][j] = float(list1[i][j])    #这里分割完了以后转float格式

        if list1[0][2] < min_time:  #如果有轨迹的起始时间比我们设置的最小时间还小，就舍弃掉
            continue

        arr = np.array(list1)   #将现在这个float格式的list转成numpy的array便于处理

        
        if arr[:,0].max()>=max0 or arr[:,0].min()<=min0 or arr[:,1].max()>=max1 or arr[:,1].min()<=min1:  #筛选掉存在超出系统区域限制的点的轨迹  
            continue
        
        arr_sorted = [] #这是用来装最后处理完的轨迹的list，为什么要单独设置一个，是因为后面要去除因为时隙化导致的冗余，用原数组arr不方便
        
        ii = int((arr[0,2]-min_time)/INTERVAL)  #计算这条轨迹起始时间对应的时隙编号

        #这一段的意思是，按轨迹中的每一个点计算：一方面把经纬度归一化到L*L的区域，一方面计算时隙编号
        #                                    如果和上一个点的时隙编号相同，就忽略这个点，否则在arr_sorted中添加这个点
        for i in range(len(arr)):
            if i == 0:
                arr_sorted.append([L*(arr[i,0]-min0)/(max0-min0), L*(arr[i,1]-min1)/(max1-min1), ii])
            elif int((arr[i,2]-min_time)/INTERVAL) != int((arr[i-1,2]-min_time)/INTERVAL):
                ii = ii+1
                arr_sorted.append([L*(arr[i,0]-min0)/(max0-min0), L*(arr[i,1]-min1)/(max1-min1), ii])
            else:
                pass

        arr_sorted = np.array(arr_sorted)   #这里arr_sorted已经处理好了，把它转成数组

        if int(arr_sorted[0][2]/4320) in range(StartDate-min_date, StartDate-min_date+Length):  #按这条轨迹起始时间处在的天分类，存在日期对应的字典里
            dict_Car[min_date+int(arr_sorted[0][2]/4320)].update({count:arr_sorted})
            count = count+1 

        # if count>5:
        #     break   

    if num > 1000:  #读取的批数的上限，超过就break
        break

# print(dict_Car)
gc.enable();

print("begin data store!")
for i in range(StartDate, StartDate+Length):
    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(i)+'.txt','w')   #把字典存在txt文件里
    f.write(str(dict_Car[i]))
    f.close()