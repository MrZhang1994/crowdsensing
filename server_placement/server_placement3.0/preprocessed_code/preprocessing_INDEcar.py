import numpy as np
from matplotlib import pyplot as plt
from numpy import array as array

f = open('../preprocessed_data/dict_Car.txt','r')
a = f.read()
dict_Car = eval(a)

count = 0
lenth = []
dict_INDECar = {}
for key in dict_Car:
    arr_sort = dict_Car[key]
    if abs(arr_sort[0,0]-arr_sort[-1,0])+abs(arr_sort[0,1]-arr_sort[-1,1])>12000 and len(arr_sort)>30:
        count = count+1
        arr = np.zeros((2*len(arr_sort),2))
        arr[0:len(arr_sort),:] = arr_sort[:,0:2]
        for i in range(len(arr_sort)):
            arr[i+len(arr_sort),:] = arr_sort[len(arr_sort)-1-i,0:2]
        dict_INDECar.update({count:arr})
        

print(count)
print(dict_INDECar)

print("begin data store!")
f = open('../preprocessed_data/dict_INDECar.txt','w')
f.write(str(dict_INDECar))
f.close()