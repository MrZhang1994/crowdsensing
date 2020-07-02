import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import array as array


'''
因为上一个文件只是得到了每一天有那些车，但是不知道每个时刻系统要进入哪些车，这个文件就是为此做的一个索引
'''

for i in range(1,31):   #从1到30号
    print(i)
    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1000+i)+'.txt','r')  #读取每一天对应的字典
    a = f.read()
    dict_Car_1001 = eval(a)
    f.close()

    dict_Car_1001_TimeIndex = {}    #这个是索引

    for key in list(dict_Car_1001.keys()):  #对字典中的每一辆车
        start_time = (dict_Car_1001[key])[0][2] #得到起始时隙编号
        if start_time in dict_Car_1001_TimeIndex.keys():    #如果起始时隙编号已经在dict_Car_1001_TimeIndex的键中
            dict_Car_1001_TimeIndex[start_time].append(key) #就在作为值的字典中加上这辆车的编号
        else:
            dict_Car_1001_TimeIndex.update({start_time:[key]})  #否则在字典dict_Car_1001_TimeIndex中加上{起始时隙编号：{车编号}}这个键值对

    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1000+i)+'_TimeIndex.txt','w')    #保存索引字典
    f.write(str(dict_Car_1001_TimeIndex))
    f.close()


# # print(dict_Car_1001_TimeIndex)
# count = np.zeros((4320,))
# for time in range(4320):
#     if time in dict_Car_1001_TimeIndex.keys():
#         count[time] = len(dict_Car_1001_TimeIndex[time])

# plt.plot(range(4320), count)

# plt.grid()
# plt.show()

# count = np.zeros((4320,))

# try:
    
#     for key in list(dict_Car_1001.keys()):
#         start_time = (dict_Car_1001[key])[0][2]
#         stop_time = (dict_Car_1001[key])[-1][2]
#         for i in range(int(start_time-2*4320), int(stop_time-2*4320+1)):
#             if i >= 4320:
#                 break
#             count[i] += 1
# except IndexError:
#     print(dict_Car_1001[key+1])
#     print(dict_Car_1001[key])
#     print('\n')



# plt.plot(range(4320), count)

# plt.grid()
# plt.show()