import numpy as np
from numpy import array as array
from matplotlib import pyplot as plt


# some parameters
L = 30 # the numbers of rows and columns
MAX_QUEUING_TIME = 1000
INDE_CAPACITY = 5 # how many cars one INDE can serve 
U_BEST = 0.7 # as the name say



class Car:
    def __init__(self, arr):
        self.arr = arr
        self.start = int(arr[0,2])
        self.pos = arr[0,0:2]
        self.INDE = -1
        self.distance = 99999
        self.delay = 99999
    def update(self, t, dict_bt_cell, rho, list_bt_object):
        self.pos = self.arr[t-self.start,0:2]
        self.update_INDE(dict_bt_cell, rho, list_bt_object)
        # if self.INDE != -1:
        #     self.delay = abs(self.pos[0]-(list_bt_object[self.INDE]).pos[0])+abs(self.pos[1]-(list_bt_object[self.INDE]).pos[1])
        
        # if self.delay > 500:
        #     '''
        #     find a new INDE
        #     '''
        #     self.update_INDE(dict_bt_cell, list_bt_object)

        # if self.INDE != -1:
        #     self.delay = abs(self.pos[0]-(bt_pos[self.INDE])[0])+abs(self.pos[1]-(bt_pos[self.INDE])[1])                  

        return self.INDE

    def calculate_delay(self, INDE_id, rho, list_bt_object):
        if INDE_id == -1:
            self.delay = 99999
        else:
            if list_bt_object[INDE_id].INDEtype == 0: # a bt INDE
                access_delay = 10
                x = int(600*list_bt_object[INDE_id].pos[0]/10000)
                y = int(600*list_bt_object[INDE_id].pos[1]/10000)
                queuing_delay = MAX_QUEUING_TIME*rho[x][y]
                self.delay = access_delay+queuing_delay
            else: # a car INDE
                self.distance = np.sqrt(np.square(self.pos[0]-list_bt_object[INDE_id].pos[0])+np.square(self.pos[1]-list_bt_object[INDE_id].pos[1]))
                if self.distance <= 400:
                    self.delay = 5
                else:
                    self.delay = 400

    def update_INDE(self, dict_bt_cell, rho, list_bt_object):
        cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
        if dict_bt_cell[cell_number]:
            # self.INDE = choice(dict_bt_cell[cell_number])
            for i in range(len(dict_bt_cell[cell_number])):
                pos0 = list_bt_object[((dict_bt_cell[cell_number])[i])].pos[0]
                pos1 = list_bt_object[((dict_bt_cell[cell_number])[i])].pos[1]
                new_distance = np.sqrt(np.square(self.pos[0]-pos0)+np.square(self.pos[1]-pos1))
                if new_distance < self.distance and list_bt_object[((dict_bt_cell[cell_number])[i])].report_load_rate()<1:
                    self.INDE = (dict_bt_cell[cell_number])[i]
                    self.distance = new_distance         
            # if self.delay >500:
            #     self.INDE = -1
        else:
            self.INDE = -1
        self.calculate_delay(self.INDE, rho, list_bt_object)
        # else:
        #     if cell_number == 0:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #     elif cell_number == 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])                    
        #     elif cell_number == 90:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number == 99:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number < 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #     elif cell_number > 90:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number%10 == 0:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     elif cell_number%10 == 9:
        #         if dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])
        #     else:
        #         if dict_bt_cell[cell_number+1]:
        #             self.INDE = choice(dict_bt_cell[cell_number+1])
        #         elif dict_bt_cell[cell_number-1]:
        #             self.INDE = choice(dict_bt_cell[cell_number-1])
        #         elif dict_bt_cell[cell_number+10]:
        #             self.INDE = choice(dict_bt_cell[cell_number+10])
        #         elif dict_bt_cell[cell_number-10]:
        #             self.INDE = choice(dict_bt_cell[cell_number-10])         

class Bt:
    def __init__(self, ID, INDEtype, arr):
        self.id = ID
        self.INDEtype = INDEtype # INDEtype=1 means it's a car; =0 means it's a bt
        self.arr = arr
        if self.INDEtype == 0:
            self.pos = arr
        else:
            self.pos = arr[0,:]
        # self.pos[0] = pos[0]
        # self.pos[1] = pos[1]
        self.state = 0
        self.ConectNum = 0
        self.timestamp = 0
        self.arrlen = len(self.arr)

    def bt_update(self, dict_bt_cell):
        
        old_cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000) 
        if self.INDEtype == 1: # means it's a carINDE
            if self.timestamp >= self.arrlen:
                self.timestamp = 0
            self.pos = self.arr[self.timestamp,:]
            self.timestamp = self.timestamp+1
        if self.state == 1:
            new_cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
            if new_cell_number != old_cell_number:
                dict_bt_cell[old_cell_number].remove(self.id)
                if new_cell_number in dict_bt_cell.keys():
                    if self.id not in dict_bt_cell[new_cell_number]:
                        dict_bt_cell[new_cell_number].append(self.id)
                    else:
                        print('dict_bt_cell append error!')
                else:
                    dict_bt_cell.update({new_cell_number:[self.id]})

    def open(self):
        self.state = 1
        '''
        add itself into open list
        '''
    def close(self):
        self.state = 0
        self.ConectNum = 0
        '''
        delete itself frome open list
        '''
    def connect(self):
        if self.state == 1:
            self.ConectNum = self.ConectNum+1
        else:
            print('the INDE is not open!')

    def disconnect(self):
        if self.ConectNum != 0:
            self.ConectNum = self.ConectNum-1
        else:
            print('no car connect to the INDE now, cannot do the disconnection')

    def report_load_rate(self):
        if self.state == 1:
            load_rate = self.ConectNum/INDE_CAPACITY
        else:
            load_rate = U_BEST
        return(load_rate)

class Env:
    '''
    about the envornment
    '''
    def __init__(self):
        '''initializing'''
        self.bt_pos = np.loadtxt('bt_inside_1910.txt')
        self.rho = np.loadtxt('rho.txt')
        f = open('dict_Car.txt','r')
        a = f.read()
        self.dict_Car = eval(a)
        f.close()
        f = open('dict_INDECar.txt','r')
        a = f.read()
        self.dict_IDNECar = eval(a)
        f.close()

        '''
        some list
        '''
        self.list_bt_object = []
        self.dict_existing_car = {}
        self.car_id = 0
        self.dict_open_bt = {}
        self.load_rate = np.zeros((len(self.bt_pos)+len(self.dict_IDNECar),1))
        self.dict_bt_cell = {}
        '''
        create object
        '''
        tempid = 0
        for i in range(len(self.bt_pos)):
            self.list_bt_object.append(Bt(tempid, 0, self.bt_pos[i,:]))
            tempid += 1
        for key in self.dict_IDNECar:
            self.list_bt_object.append(Bt(tempid, 1, self.dict_IDNECar[key]))
            tempid += 1

            # cell_number = 10*int(self.bt_pos[i,0]/(10000/L))+int(self.bt_pos[i,1]/(10000/L))
            # if self.dict_bt_cell[cell_number]:
            #     self.dict_bt_cell[cell_number].append(i)
            # else:
            #     self.dict_bt_cell.update({dict_bt_cell[cell_number]:[i]})
        # for i in range(20):
        #     self.dict_existing_car.update({self.car_id:Car(self.dict_Car[i])})
        #     self.car_id = self.car_id+1

        # figure
        # plt.ion()


    def update(self, a, a_last, t):
        '''
        an updation through time
        '''
        # update BTs' openning state through action a
        for i in range(len(self.list_bt_object)):
            self.list_bt_object[i].bt_update(self.dict_bt_cell)
            if a[i] == 1 and a_last[i] == 0:
                self.list_bt_object[i].open()
                self.dict_open_bt.update({i:[]})
                cell_number = 10*int(self.list_bt_object[i].pos[0]/1000)+int(self.list_bt_object[i].pos[1]/1000)
                if cell_number in self.dict_bt_cell.keys():
                    if i not in self.dict_bt_cell[cell_number]:
                        self.dict_bt_cell[cell_number].append(i)
                else:
                    self.dict_bt_cell.update({cell_number:[i]})  

            elif a[i] == 0 and a_last[i] == 1:
                self.list_bt_object[i].close()
                cell_number = 10*int(self.list_bt_object[i].pos[0]/1000)+int(self.list_bt_object[i].pos[1]/1000)
                # print(i)
                self.dict_bt_cell[cell_number].remove(i)

                if self.dict_open_bt[i]:
                    for j in range(len(self.dict_open_bt[i])):
                        # print(self.dict_open_bt[i][j])
                        self.dict_existing_car[self.dict_open_bt[i][j]].INDE = -1
                        self.dict_existing_car[self.dict_open_bt[i][j]].delay = 99999
                del self.dict_open_bt[i]

        # add new car to the environment
        for i in range(20):
            if (i+20*t) in self.dict_Car.keys():
                self.dict_existing_car.update({self.car_id:Car(self.dict_Car[i+20*t])})
                self.car_id = self.car_id+1
            else:
                break

        # cars update their position and maybe find a new INDE

        for key in self.dict_existing_car:
            if self.dict_existing_car[key].INDE != -1:
                # print(self.dict_existing_car[key].INDE)
                self.list_bt_object[self.dict_existing_car[key].INDE].disconnect()
                self.dict_open_bt[self.dict_existing_car[key].INDE].remove(key)
            chosen_INDE = self.dict_existing_car[key].update(t, self.dict_bt_cell, self.rho, self.list_bt_object)
            if chosen_INDE != -1:
                self.list_bt_object[chosen_INDE].connect()
                self.dict_open_bt[chosen_INDE].append(key)

        # calculate load rate and create system log
        '''
        PrintLogType = 0 means print all INDE
        PrintLogType = 1 means print connected INDE
        PrintLogType = 2 means print open INDE
        '''
        PrintLogType = 1
        if t == 0:
            f = open('log'+str(PrintLogType)+'.txt', 'w')
        else:
            f = open('log'+str(PrintLogType)+'.txt', 'a')
        f.write('================================ t = '+str(t)+' =======================================\n'+'\n')
        f.close()
        if PrintLogType == 0:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                f = open('log'+str(PrintLogType)+'.txt', 'a')
                log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                log += 'loadrate='+str(self.load_rate[i])+'||'
                log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                f.write(log)
                if i in self.dict_open_bt.keys(): 
                    if self.dict_open_bt[i]:
                        f.write('connecting cars information:\n')
                        for j in range(len(self.dict_open_bt[i])):
                            log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                            log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                            log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                            log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                            f.write(log)
                f.write('\n')           
                f.close()
        elif PrintLogType == 1:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                if self.list_bt_object[i].ConectNum != 0:
                    f = open('log'+str(PrintLogType)+'.txt', 'a')
                    log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                    log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                    log += 'loadrate='+str(self.load_rate[i])+'||'
                    log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                    log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                    f.write(log)
                    if i in self.dict_open_bt.keys(): 
                        if self.dict_open_bt[i]:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.dict_open_bt[i])):
                                log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                                f.write(log)
                    f.write('\n')           
                    f.close()            
        elif PrintLogType == 2:
            for i in range(len(self.list_bt_object)):
                self.load_rate[i] = self.list_bt_object[i].report_load_rate()
                # create system log
                if self.list_bt_object[i].state != 0:
                    f = open('log'+str(PrintLogType)+'.txt', 'a')
                    log = 'INDE='+str(i)+'||state='+str(self.list_bt_object[i].state)+'||'
                    log += 'INDEtype='+str(self.list_bt_object[i].INDEtype)+'||'
                    log += 'loadrate='+str(self.load_rate[i])+'||'
                    log += 'pos='+str(self.list_bt_object[i].pos)+'||'
                    log += 'ConectNum='+str(self.list_bt_object[i].ConectNum)+'\n'
                    f.write(log)
                    if i in self.dict_open_bt.keys(): 
                        if self.dict_open_bt[i]:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.dict_open_bt[i])):
                                log = 'carID='+str(self.dict_open_bt[i][j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.dict_open_bt[i][j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.dict_open_bt[i][j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.dict_open_bt[i][j]].delay)+'\n'
                                f.write(log)
                    f.write('\n')          
                    f.close()
        f = open('log'+str(PrintLogType)+'.txt', 'a')
        f.write('Cars that dont have a INDE:'+'\n')
        for key in list(self.dict_existing_car.keys()):
            if self.dict_existing_car[key].INDE == -1:
                log = 'carID='+str(key)+'||'+'pos='+str(self.dict_existing_car[key].pos)+'||'+'delay='+str(self.dict_existing_car[key].delay)+'\n'
                f.write(log)
        f.write('\n')
        f.close()

        # del cars
        for key in list(self.dict_existing_car.keys()):
            # if t == self.dict_existing_car[key].arr[-1,2]:
            if t == self.dict_existing_car[key].arr[-1,2]:
                if self.dict_existing_car[key].INDE != -1:
                    self.list_bt_object[self.dict_existing_car[key].INDE].disconnect()
                    (self.dict_open_bt[self.dict_existing_car[key].INDE]).remove(key)
                del self.dict_existing_car[key]

        # # draw picture
        # cars = []
        # for key in self.dict_existing_car:
        #     cars.append([self.dict_existing_car[key].pos[0],self.dict_existing_car[key].pos[1]])
        # cars = np.array(cars)
        # bts = []
        # for key in self.dict_open_bt:
        #     bts.append([self.list_bt_object[key].pos[0],self.list_bt_object[key].pos[1]])
        # bts = np.array(bts)
        # # print(bts)
        # plt.clf()
        # if len(cars)>0:
        #     plt.scatter(cars[:,0], cars[:,1], c = 'b', marker = '.')
        # if len(bts)>0:
        #     plt.scatter(bts[:,0], bts[:,1], c = 'r', marker = '.')
        # plt.xlim(0, 10000)
        # plt.ylim(0, 10000)
        # plt.show()
        # plt.savefig('result_images/'+str(t)+'.jpg')
        # plt.pause(0.2)
        # # plt.ioff()
        
        return self.load_rate

