import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy import array as array

for i in range(1,11):
    print(i)
    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1000+i)+'.txt','r')
    a = f.read()
    dict_Car_1001 = eval(a)
    f.close()

    dict_Car_1001_TimeIndex = {}

    for key in list(dict_Car_1001.keys()):
        start_time = (dict_Car_1001[key])[0][2]
        if start_time in dict_Car_1001_TimeIndex.keys():
            dict_Car_1001_TimeIndex[start_time].append(key)
        else:
            dict_Car_1001_TimeIndex.update({start_time:[key]})

    f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1000+i)+'_TimeIndex.txt','w')
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