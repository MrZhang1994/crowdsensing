import numpy as np
from numpy import array as array
from matplotlib import pyplot as plt


# some parameters
L = 30 # the numbers of rows and columns
MAX_QUEUING_TIME = 1000
L1_INDE_CAPACITY = 5 # how many cars one INDE can serve 
L2_INDE_CAPACITY = 10
U_BEST = 0.7 # as the name say

# control mode
PrintLogType = -1
PictureType = 1
LogLocation = 'log/'

a_dim = 1910
s_dim = 1910*3

class Car:
    def __init__(self, ID, arr):
        self.id = ID
        self.arr = arr
        self.start = int(arr[0,2])
        self.pos = arr[0,0:2]
        self.distance = 99999
        self.delay = 99999
        self.INDE = -1

    def update(self, t, dict_L1_INDE_cell, rho, list_INDE_object):
        self.pos = self.arr[t-self.start,0:2]
        if self.INDE != -1:
            list_INDE_object[self.INDE].L1_disconnect_Car(self.id)
        self.update_INDE(dict_L1_INDE_cell, rho, list_INDE_object)
        if self.INDE != -1:
            list_INDE_object[self.INDE].L1_connect_Car(self.id)

    def calculate_delay(self, INDE_id, rho, list_INDE_object):
        if INDE_id == -1:
            self.delay = 99999
        else:
            if list_INDE_object[INDE_id].INDEtype == 0: # a INDE INDE
                access_delay = 10
                x = int(600*list_INDE_object[INDE_id].pos[0]/10000)
                y = int(600*list_INDE_object[INDE_id].pos[1]/10000)
                queuing_delay = MAX_QUEUING_TIME*rho[x][y]
                self.delay = access_delay+queuing_delay
            else: # a car INDE
                self.distance = np.sqrt(np.square(self.pos[0]-list_INDE_object[INDE_id].pos[0])+np.square(self.pos[1]-list_INDE_object[INDE_id].pos[1]))
                if self.distance <= 400:
                    self.delay = 5
                else:
                    self.delay = 400

    def update_INDE(self, dict_L1_INDE_cell, rho, list_INDE_object):
        cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
        if dict_L1_INDE_cell[cell_number]:
            # self.INDE = choice(dict_L1_INDE_cell[cell_number])
            for i in range(len(dict_L1_INDE_cell[cell_number])):
                pos0 = list_INDE_object[((dict_L1_INDE_cell[cell_number])[i])].pos[0]
                pos1 = list_INDE_object[((dict_L1_INDE_cell[cell_number])[i])].pos[1]
                new_distance = np.sqrt(np.square(self.pos[0]-pos0)+np.square(self.pos[1]-pos1))
                if new_distance < self.distance and list_INDE_object[((dict_L1_INDE_cell[cell_number])[i])].L1_report_loadrate()<1:
                    self.INDE = (dict_L1_INDE_cell[cell_number])[i]
                    self.distance = new_distance         
        else:
            self.INDE = -1
        self.calculate_delay(self.INDE, rho, list_INDE_object)
       
class INDE:
    def __init__(self, ID, INDEtype, arr):
        # Hardware attribute
        self.id = ID
        self.INDEtype = INDEtype # INDEtype=1 means it's a car; =0 means it's a INDE
        self.arr = arr
        if self.INDEtype == 0:
            self.pos = arr
        else:
            self.pos = arr[0,:]
        self.timestamp = 0
        self.arrlen = len(self.arr)

        # L1 software attribute
        self.L1_state = 0
        self.L1_connecting_car_list = []
        self.L1_delay = 999999
        self.L1_parentIDNE = -1
        self.L1_loadrate = 0

        #L2 software attribute
        self.L2_state = 0
        self.L2_connecting_L1_list = []
        self.L2_loadrate = 0



    def INDE_pos_update(self, dict_L1_INDE_cell):
        
        old_cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000) 
        if self.INDEtype == 1: # means it's a carINDE
            if self.timestamp >= self.arrlen:
                self.timestamp = 0
            self.pos = self.arr[self.timestamp,:]
            self.timestamp = self.timestamp+1
        if self.L1_state == 1 or self.L2_state == 1:
            new_cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
            if new_cell_number != old_cell_number:
                dict_L1_INDE_cell[old_cell_number].remove(self.id)
                if new_cell_number in dict_L1_INDE_cell.keys():
                    if self.id not in dict_L1_INDE_cell[new_cell_number]:
                        dict_L1_INDE_cell[new_cell_number].append(self.id)
                    else:
                        print('dict_L1_INDE_cell append error!')
                else:
                    dict_L1_INDE_cell.update({new_cell_number:[self.id]})

    # Some operations on the first layer
    def L1_open(self, list_open_L1_INDE, dict_L1_INDE_cell):
        self.L1_state = 1
        self.L1_connecting_car_list = []
        self.L1_delay = 999999
        self.L1_parentIDNE = -1
        self.L1_loadrate = 0
        '''
        add itself into open list
        '''
        list_open_L1_INDE.append(self.id)
        cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
        if cell_number in dict_L1_INDE_cell.keys():
            if self.id not in dict_L1_INDE_cell[cell_number]:
                dict_L1_INDE_cell[cell_number].append(self.id)
        else:
            dict_L1_INDE_cell.update({cell_number:[self.id]})

    def L1_close(self, list_open_L1_INDE, dict_L1_INDE_cell, dict_existing_car):
        self.L1_state = 0
        self.L1_delay = 999999
        self.L1_parentIDNE = -1
        self.L1_loadrate = 0
        '''
        delete itself frome open list
        '''
        for i in range(len(self.L1_connecting_car_list)):
            dict_existing_car[self.L1_connecting_car_list[i]].INDE = -1
            dict_existing_car[self.L1_connecting_car_list[i]].delay = 99999   
        self.L1_connecting_car_list = []
        cell_number = 10*int(self.pos[0]/1000)+int(self.pos[1]/1000)
        dict_L1_INDE_cell[cell_number].remove(self.id)
        list_open_L1_INDE.remove(self.id)
        

    def L1_connect_Car(self, car_id):
        if self.L1_state == 1:
            self.L1_connecting_car_list.append(car_id)
            self.L1_loadrate = len(self.L1_connecting_car_list)/L1_INDE_CAPACITY
        else:
            print('the L1 INDE is not open!')

    def L1_disconnect_Car(self, car_id):
        if self.L1_connecting_car_list:
            self.L1_connecting_car_list.remove(car_id)
            self.L1_loadrate = len(self.L1_connecting_car_list)/L1_INDE_CAPACITY
        else:
            print('no car connect to the INDE now, cannot do the disconnection')

    def L1_report_loadrate(self):
        return(self.L1_loadrate)

    # Some operations on the second layer
    def L2_open(self):
        self.L2_state = 1
        self.L2_connecting_L1_list = []
        '''
        add itself into open list
        '''
        list_open_L2_INDE.append(self.id)

    def L2_close(self, list_INDE_object):
        self.L2_state = 0
        for i in range(len(self.L2_connecting_L1_list)):
            list_INDE_object[self.L2_connecting_L1_list[i]].L1_parentIDNE = -1
            list_INDE_object[self.L2_connecting_L1_list[i]].delay = 999999
        self.L2_connecting_L1_list = []
        list_open_L2_INDE.remove(self.id)

    def L2_connect_L1(self, INDE_id):
        if self.L2_state == 1:
            self.L2_connecting_L1_list.append(INDE_id)
            self.L2_loadrate = len(self.L2_connecting_L1_list)/L2_INDE_CAPACITY
        else:
            print('the L2 INDE is not open!')

    def L2_disconect_L1(self, INDE_id):
        if self.L2_connecting_L1_list:
            self.L2_connecting_L1_list.remove(INDE_id)
            self.L2_loadrate = len(self.L2_connecting_L1_list)/L2_INDE_CAPACITY
        else:
            print('no L1 connect to the L2 now, cannot do the disconnection')

    def L2_report_loadrate(self):
        return(self.L2_loadrate)

    def Render(self):
        # L1 software attribute
        # self.L1_state = 0
        self.L1_connecting_car_list = []
        self.L1_delay = 999999
        self.L1_parentIDNE = -1
        self.L1_loadrate = 0

        #L2 software attribute
        # self.L2_state = 0
        self.L2_connecting_L1_list = []
        self.L2_loadrate = 0  

class Env:
    '''
    about the envornment
    '''
    def __init__(self):
        '''initializing'''
        self.INDE_pos = np.loadtxt('bt_inside_1910.txt')
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
        self.list_INDE_object = []
        self.dict_existing_car = {}
        self.car_counter = 0 # a numbering counter

        self.dict_L1_INDE_cell = {}

        self.list_open_L1_INDE = []
        self.list_open_L2_INDE = []

        self.load_rate = np.zeros((a_dim,1))

        '''
        create object
        '''
        tempid = 0
        for i in range(len(self.INDE_pos)):
            self.list_INDE_object.append(INDE(tempid, 0, self.INDE_pos[i,:]))
            tempid += 1
        # for key in self.dict_IDNECar:
        #     self.list_INDE_object.append(INDE(tempid, 1, self.dict_IDNECar[key]))
        #     tempid += 1

        if PictureType != -1:
            plt.ion()


    def update(self, a, a_last, t):
        '''
        an updation through time
        '''
        # update INDEs' openning state through action a
        for i in range(len(self.list_INDE_object)):
            self.list_INDE_object[i].INDE_pos_update(self.dict_L1_INDE_cell)
            if a[i] == 1 and a_last[i] == 0:
                self.list_INDE_object[i].L1_open(self.list_open_L1_INDE, self.dict_L1_INDE_cell)
            elif a[i] == 0 and a_last[i] == 1:
                self.list_INDE_object[i].L1_close(self.list_open_L1_INDE, self.dict_L1_INDE_cell, self.dict_existing_car)

        # add new car to the environment
        for i in range(20):
            if (i+20*t) in self.dict_Car.keys():
                self.dict_existing_car.update({self.car_counter:Car(self.car_counter, self.dict_Car[i+20*t])})
                self.car_counter = self.car_counter+1
            else:
                break

        # cars update their position and maybe find a new INDE

        for key in self.dict_existing_car:
            self.dict_existing_car[key].update(t, self.dict_L1_INDE_cell, self.rho, self.list_INDE_object)

        # calculate load rate and create system log
        '''
        PrintLogType = 0 means print all INDE
        PrintLogType = 1 means print connected INDE
        PrintLogType = 2 means print open INDE
        '''
        self.Print_L1_Car_Log(t, PrintLogType)

        # del cars
        for key in list(self.dict_existing_car.keys()):
            # if t == self.dict_existing_car[key].arr[-1,2]:
            if t == self.dict_existing_car[key].arr[-1,2]:
                if self.dict_existing_car[key].INDE != -1:
                    self.list_INDE_object[self.dict_existing_car[key].INDE].L1_disconnect_Car(key)
                del self.dict_existing_car[key]

        # # draw picture
        self.Draw_L1_Car_Picture(t, PictureType)
        
        return np.vstack((self.load_rate, self.GetPos()))

    def Print_L1_Car_Log(self, t, PrintLogType):
        '''
        calculate load rate and create system log:
        PrintLogType = 0 means print all INDE
        PrintLogType = 1 means print connected INDE
        PrintLogType = 2 means print open INDE
        PrintLogType = -1 means do nothing
        '''
        if PrintLogType != -1:
            if t == 0:
                f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'w')
            else:
                f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'a')
            f.write('================================ t = '+str(t)+' =======================================\n'+'\n')
            f.close()
            if PrintLogType == 0:
                for i in range(len(self.list_INDE_object)):
                    self.load_rate[i] = self.list_INDE_object[i].L1_report_loadrate()
                    # create system log
                    f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'a')
                    log = 'INDE='+str(i)+'||state='+str(self.list_INDE_object[i].L1_state)+'||'
                    log += 'INDEtype='+str(self.list_INDE_object[i].INDEtype)+'||'
                    log += 'loadrate='+str(self.load_rate[i])+'||'
                    log += 'pos='+str(self.list_INDE_object[i].pos)+'||'
                    log += 'ConectNum='+str(len(self.list_INDE_object[i].L1_connecting_car_list))+'\n'
                    f.write(log)
                    if self.list_INDE_object[i].L1_connecting_car_list:
                        f.write('connecting cars information:\n')
                        for j in range(len(self.list_INDE_object[i].L1_connecting_car_list)):
                            log = 'carID='+str(self.list_INDE_object[i].L1_connecting_car_list[j])+'||'
                            log += 'carPOS='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].pos)+'||'
                            log += 'carINDE='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].INDE)+'||'
                            log += 'carDELAY='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].delay)+'\n'
                            f.write(log)
                    f.write('\n')           
                    f.close()
            elif PrintLogType == 1:
                for i in range(len(self.list_INDE_object)):
                    self.load_rate[i] = self.list_INDE_object[i].L1_report_loadrate()
                    # create system log
                    if len(self.list_INDE_object[i].L1_connecting_car_list) != 0:
                        f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'a')
                        log = 'INDE='+str(i)+'||state='+str(self.list_INDE_object[i].L1_state)+'||'
                        log += 'INDEtype='+str(self.list_INDE_object[i].INDEtype)+'||'
                        log += 'loadrate='+str(self.load_rate[i])+'||'
                        log += 'pos='+str(self.list_INDE_object[i].pos)+'||'
                        log += 'ConectNum='+str(len(self.list_INDE_object[i].L1_connecting_car_list))+'\n'
                        f.write(log)
                        if self.list_INDE_object[i].L1_connecting_car_list:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.list_INDE_object[i].L1_connecting_car_list)):
                                log = 'carID='+str(self.list_INDE_object[i].L1_connecting_car_list[j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].delay)+'\n'
                                f.write(log)
                        f.write('\n')           
                        f.close()            
            elif PrintLogType == 2:
                for i in range(len(self.list_INDE_object)):
                    self.load_rate[i] = self.list_INDE_object[i].L1_report_loadrate()
                    # create system log
                    if self.list_INDE_object[i].L1_state != 0:
                        f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'a')
                        log = 'INDE='+str(i)+'||state='+str(self.list_INDE_object[i].L1_state)+'||'
                        log += 'INDEtype='+str(self.list_INDE_object[i].INDEtype)+'||'
                        log += 'loadrate='+str(self.load_rate[i])+'||'
                        log += 'pos='+str(self.list_INDE_object[i].pos)+'||'
                        log += 'ConectNum='+str(len(self.list_INDE_object[i].L1_connecting_car_list))+'\n'
                        f.write(log)
                        if self.list_INDE_object[i].L1_connecting_car_list:
                            f.write('connecting cars information:\n')
                            for j in range(len(self.list_INDE_object[i].L1_connecting_car_list)):
                                log = 'carID='+str(self.list_INDE_object[i].L1_connecting_car_list[j])+'||'
                                log += 'carPOS='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].pos)+'||'
                                log += 'carINDE='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].INDE)+'||'
                                log += 'carDELAY='+str(self.dict_existing_car[self.list_INDE_object[i].L1_connecting_car_list[j]].delay)+'\n'
                                f.write(log)
                        f.write('\n')          
                        f.close()
            f = open(LogLocation+'log'+str(PrintLogType)+'.txt', 'a')
            f.write('Cars that dont have a INDE:'+'\n')
            for key in list(self.dict_existing_car.keys()):
                if self.dict_existing_car[key].INDE == -1:
                    log = 'carID='+str(key)+'||'+'pos='+str(self.dict_existing_car[key].pos)+'||'+'delay='+str(self.dict_existing_car[key].delay)+'\n'
                    f.write(log)
            f.write('\n')
            f.close()
        else:
            for i in range(len(self.list_INDE_object)):
                self.load_rate[i] = self.list_INDE_object[i].L1_report_loadrate()

    def Draw_L1_Car_Picture(self, t, PictureType):
        '''
        Draw pictures of INDEs and cars:
        PictureType = 0 means only save pictures
        PictureType = 1 means show and save pictures
        PictureType = -1 means do nothing
        '''
        if PictureType != -1:
            cars = []
            for key in self.dict_existing_car:
                cars.append([self.dict_existing_car[key].pos[0],self.dict_existing_car[key].pos[1]])
            cars = np.array(cars)
            INDEs = []
            for i in range(len(self.list_open_L1_INDE)):
                INDEs.append([self.list_INDE_object[self.list_open_L1_INDE[i]].pos[0],self.list_INDE_object[self.list_open_L1_INDE[i]].pos[1]])
            INDEs = np.array(INDEs)
            plt.clf()
            if len(cars)>0:
                plt.scatter(cars[:,0], cars[:,1], c = 'b', marker = '.')
            if len(INDEs)>0:
                plt.scatter(INDEs[:,0], INDEs[:,1], c = 'r', marker = '.')
            plt.xlim(0, 10000)
            plt.ylim(0, 10000)
            if PictureType == 0:
                plt.savefig('result_images/'+str(t)+'.jpg')
            elif PictureType == 1:
                plt.show()
                plt.savefig('result_images/'+str(t)+'.jpg')
                plt.pause(0.05)
                # plt.ioff()

    def Print_System_Performance(self):
        connected_car_count = 0
        unconnected_car_count = 0
        for key in self.dict_existing_car.keys():
            if self.dict_existing_car[key].INDE == -1:
                unconnected_car_count += 1
            else:
                connected_car_count += 1
        car_num = len(self.dict_existing_car)
        connected_car_rate = connected_car_count/car_num
        unconnected_car_rate = unconnected_car_count/car_num
        return [car_num, connected_car_rate]

    def GetPos(self):
        return self.INDE_pos.reshape((1910*2,1))
            
    def Render(self):
        for i in range(len(self.list_INDE_object)):
            self.list_INDE_object[i].Render()

        self.dict_existing_car = {}
        self.car_counter = 0 # a numbering counter

        # self.load_rate = np.zeros((a_dim,1))        