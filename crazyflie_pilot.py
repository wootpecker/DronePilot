import cflib.crtp
import logs.logger as logger
import logging
import flightpaths
import parameters
import sys
import time
from threading import Event
import pandas as pd
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

LOGS_SAVE = True
#URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]
deck_attached_event = Event()
INITIAL_POSITION = [0, 0, 0]
POSITION_ESTIMATE = [0, 0, 0]
GAS_DISTRIBUTION = [0, 0, 0]
DATASET_FLIGHTPATH = []
DATASET_COMPLETE = []
FILENAME="GSL"
START_TIME = False
START_TIME_FLIGHTPATH = None
FILEPATH="data/test.csv"



def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_test_pilot")
    URI=parameters.choose_model()
    flightpath=parameters.choose_flightpath()
    flightheight=parameters.choose_flightheight()
    distance=parameters.choose_distance()
    crazyflie_take_measurements(URI=URI, flightpath=flightpath, flightheight=flightheight, distance=distance)
    save_dataset_to_csv()
    print("DONE")

def crazyflie_take_measurements(URI=uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703'), flightpath="Snake", flightheight=50, distance=5):
    logging.info("Crazyflie takes measurements.")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        #scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
        #                                 cb=param_deck_flow)
        #time.sleep(2)

        scf.cf.param.add_update_callback(group='deck', name=f'bcFlow2',
                                             cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=100)
        logconf.add_variable('stateEstimate.x', 'float')#kalman.stateX
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('range.zrange', 'uint16_t')
        logconf.add_variable('sgp30.value1L', 'uint16_t')   #configuration_data.py
        logconf.add_variable('sgp30.value1R', 'uint16_t')        
        #set_initial_position(scf, 0, 0, 0)    x = data['kalman.stateX']


        scf.cf.log.add_config(logconf)
        global FILENAME
        FILENAME=f"GSL_{flightpath}_{flightheight}"
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            logging.error('[EXIT] No flow deck detected!')
            print('[EXIT] No flow deck detected!')
            sys.exit(1)

        #scf.cf.param.set_value('kalman.resetEstimation', '1')
        #time.sleep(1)
        #scf.cf.param.set_value('kalman.resetEstimation', '0') 
        #time.sleep(1)
        window_shape=init_windowshape(logconf)
        coordinates=flightpaths.flightpath_to_coordinates(flightpath=flightpath,window_shape=[20,20],pad=1,distance=distance)
        logconf.start()    
        #time.sleep(1)
        #scf=set_initial_position(scf,flightheight)
        #dynamic_plot.dynamic_plot(flightpath=flightpath,window_size=[100,100])
        #time.sleep(2)
        cf = scf.cf
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)
        if(flightpath=="Nothing"):
            print("Flightpath: Nothing")
            time.sleep(20)
            logconf.stop()
            return
        if(flightpath=="StartLand"):
            print("Flightpath: StartLand")            
            fly_start_land(scf, flightheight)
        elif(flightpath=="TestPositioning"):
            print("Flightpath: TestPositioning")            
            fly_position_pattern(scf,flightheight)        
        elif(flightpath=="Snake"):
            print("Flightpath: Snake")                        
            fly_snake_pattern(scf, flightheight, coordinates)
        elif(flightpath=="Cage"):
            print("Flightpath: Cage")                        
            fly_cage_pattern(scf, flightheight, coordinates)

        #take_off_simple(scf)
        # move_linear_simple(scf)
        # move_box_limit(scf)
        time.sleep(2)
        logconf.stop()



def fly_position_pattern(scf, flightheight):

    print(f"INITIAL_POSITION: {INITIAL_POSITION}")
    print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")

    with MotionCommander(scf, default_height=flightheight) as mc:
        fly_take_off(mc,flightheight)
        mc.stop()      
        set_starting_position()
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        time.sleep(3)
        print("0.3")
        mc.move_distance(0.3,0.0,0)
        time.sleep(2)
        print("0.6")
        mc.move_distance(0.0,0.3,0)
        time.sleep(2)
        fly_landing_position(mc)
        return
        print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
        print(f"difference: {difference}")
        mc.move_distance(difference[0],difference[1],0)
        time.sleep(2)



def fly_snake_pattern(scf, flightheight, coordinates):
    global START_TIME
    count_x=0
    count_y=0
    relative_positions=[]
    coordinates.insert(0,[0,0])
    for x in range(len(coordinates)-1):
        relative_positions.append([coordinates[x+1][0]-coordinates[x][0],coordinates[x+1][1]-coordinates[x][1]])
        if(relative_positions[x][1]):
            print("turn")
            count_y+=1
            print(f"count_y:{count_y}{relative_positions[x]}")
            count_x=0
        else:
            count_x+=1
            print(f"count_x:{count_x}{relative_positions[x]}")
            count_y=0

    #print(relative_positions)
    with MotionCommander(scf, default_height=flightheight) as mc:
        fly_take_off(mc,flightheight)      
        time.sleep(0.6) 
        starting_position=relative_positions.pop(0)
        mc.move_distance(starting_position[0],starting_position[1],0)
        START_TIME = True  
        time.sleep(0.6) 
        for koordinate in relative_positions:
            mc.move_distance(koordinate[0],koordinate[1],0) 
            #time.sleep(1.5)
            if(koordinate[1]):
                print("turn")
        START_TIME = False
       # print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        #print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        fly_landing_position(mc)  
        time.sleep(2)        
        mc.stop()
        time.sleep(2)   


def fly_snake_pattern2(cf, flightheight, coordinates):

    #cf = scf.cf
    global START_TIME

    relative_positions=[]
    coordinates.insert(0,[0,0])
    for x in range(len(coordinates)-1):
        relative_positions.append([coordinates[x+1][0]-coordinates[x][0],coordinates[x+1][1]-coordinates[x][1]])
    starting_position=relative_positions.pop(0)
    for y in range(int(flightheight*100)):
        cf.commander.send_hover_setpoint(0, 0, 0, y / 100)
        time.sleep(0.1)
    START_TIME = True
    for _ in range(20):
        cf.commander.send_hover_setpoint(starting_position[0],starting_position[1],0,0.3)
        time.sleep(0.1)
    #time.sleep(0.5)        
    for koordinate in relative_positions:
        for x in range(10):
            cf.commander.send_hover_setpoint(koordinate[0],koordinate[1], 0, flightheight)
            time.sleep(0.1)

        #time.sleep(0.2)
        if(koordinate[1]):
            print("turn")
    START_TIME = False
       # print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        #print(f"INITIAL_POSITION: {INITIAL_POSITION}")
    fly_landing_position_test(cf,flightheight)  
  
def fly_landing_position_test(cf,flightheight):
    print("Landing")
    difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]  
    while(abs(difference[0])>0.1 or abs(difference[1])>0.1):
        print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        print(f"difference: {difference}")          
        cf.send_hover_setpoint(difference[0],difference[1],0,flightheight / 100)
        time.sleep(0.5)   
        difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
             


def fly_cage_pattern(scf, flightheight, coordinates):
    global START_TIME
    relative_positions=[]
    coordinates.insert(0,[0,0])
    for x in range(len(coordinates)-1):
        relative_positions.append([coordinates[x+1][0]-coordinates[x][0],coordinates[x+1][1]-coordinates[x][1]])
    print(relative_positions)
    with MotionCommander(scf, default_height=flightheight) as mc:
        fly_take_off(mc,flightheight)        
        time.sleep(0.6) 
        starting_position=relative_positions.pop(0)
        mc.move_distance(starting_position[0],starting_position[1],0)
        START_TIME = True  
        time.sleep(0.6) 
  
        for koordinate in relative_positions:
            mc.move_distance(koordinate[0],koordinate[1],0)
            #time.sleep(0.1)
        START_TIME = False

        fly_landing_position(mc)    
       # difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
       # mc.move_distance(difference[0],difference[1],0)
        time.sleep(2)        
        mc.stop()
        time.sleep(2)        



    #for koordinate in coordinates:

def fly_landing_position(mc):
    print("Landing")
    difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]  
    while(abs(difference[0])>0.1 or abs(difference[1])>0.1):
        print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        print(f"difference: {difference}")          
        mc.move_distance(difference[0],difference[1],0)
        time.sleep(0.5)   
        difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]

      
        
def fly_take_off(mc,flightheight):    
    mc.stop()
    time.sleep(0.5)
    mc._cf.commander.send_hover_setpoint(0, 0, 0, flightheight)
    time.sleep(2)
    difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
    print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
    print(f"INITIAL_POSITION: {INITIAL_POSITION}")
    print(f"difference: {difference}")
    set_starting_position()     
   

def fly_start_land(scf, flightheight):
    with MotionCommander(scf, default_height=flightheight) as mc:
        print("Take off START")
        time.sleep(5)
        print("Take off Hover")
        
        mc.stop()
        time.sleep(5)
        print("Take off Land")

def set_starting_position():
    global INITIAL_POSITION
    INITIAL_POSITION=[POSITION_ESTIMATE[0],POSITION_ESTIMATE[1],0]


def fly_landing(scf, flightheight):
    with MotionCommander(scf, default_height=flightheight) as mc:
        print("Landing START")
        time.sleep(3)
        print("Landing END")        
        mc.stop()
        time.sleep(3)
        print("mc.stop")
        time.sleep(5)

def reset_log():
    global START_TIME_FLIGHTPATH, GAS_DISTRIBUTION, DATASET_FLIGHTPATH
    START_TIME_FLIGHTPATH = None
    DATASET_FLIGHTPATH = []


def param_deck_flow(_, value_str):
    value = int(value_str)
    logging.info(f"Deck Flow Value: {value}")
    if value:
        deck_attached_event.set()
        logging.info(f"Flow Deck is attached!")        
    else:
        logging.info(f"[EXIT] No Flow Deck detected or attached!")
        print(f"[EXIT] No Flow Deck detected or attached!")
    #return
        #sys.exit(1)


def log_pos_callback(timestamp, data, logconf):
    global POSITION_ESTIMATE, START_TIME, DATASET_FLIGHTPATH       
    POSITION_ESTIMATE[0] = data['stateEstimate.x']
    POSITION_ESTIMATE[1] = data['stateEstimate.y']
    POSITION_ESTIMATE[2] = data['stateEstimate.z']
    #GAS_DISTRIBUTION[1] = data['sgp30.value1L']
    #GAS_DISTRIBUTION[2] = data['sgp30.value1R']

    add_dataset_async(timestamp, data)
    if START_TIME:
        DATASET_FLIGHTPATH.append([timestamp, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'), data['sgp30.value1L'],data['sgp30.value1R']])
    

    # Save data to CSV asynchronously to avoid slowing down the callback
    #save_data_async(timestamp - START_TIME, POSITION_ESTIMATE, GAS_DISTRIBUTION)


executor = ThreadPoolExecutor(max_workers=1)

def add_dataset_async(t, data):
    executor.submit(save_to_list, t, data)

def save_to_list(t, data):
    global DATASET_COMPLETE
    DATASET_COMPLETE.append([t, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'),data['sgp30.value1L'],data['sgp30.value1R']])



#def set_initial_position(scf,flightheight):
  #  scf.cf.param.set_value('kalman.initialX', INITIAL_POSITION[0])
  #  scf.cf.param.set_value('kalman.initialY', INITIAL_POSITION[1])
 #   scf.cf.param.set_value('kalman.initialZ', flightheight)
  #  return scf

def init_windowshape(logconf):
    #logconf.start()
    #global POSITION_ESTIMATE
    #end_x,end_y=POSITION_ESTIMATE[0],POSITION_ESTIMATE[1]
    #logconf.stop()
    pass
    




def save_to_csv(t, position_estimate, gas_distribution):
    with open(FILEPATH, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow([t, format(position_estimate[0], '.10f'), format(position_estimate[1], '.10f'), format(position_estimate[2], '.10f'),
                            format(position_estimate[0] - INITIAL_POSITION[0], '.10f'), format(position_estimate[1] - INITIAL_POSITION[1], '.10f'),
                            format(position_estimate[2] - INITIAL_POSITION[2], '.10f'), gas_distribution[0], gas_distribution[1], gas_distribution[2]])



def save_dataset_to_csv():
    target_dir_path = Path("data")
    target_dir_path.mkdir(parents=True, exist_ok=True) 
    if  DATASET_FLIGHTPATH:
        file_path = target_dir_path / f"{FILENAME}_F.csv"
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['Time', 'X', 'Y', 'Z', 'Gas1L', 'Gas1R'])
            start_time = DATASET_FLIGHTPATH[0][0]            
            for data in DATASET_FLIGHTPATH:
                data[0] -= start_time
                csvwriter.writerow(data)

    file_path = target_dir_path / f"{FILENAME}_C.csv"
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Time', 'X', 'Y', 'Z', 'Gas1L', 'Gas1R'])
        start_time = DATASET_COMPLETE[0][0]            
        for data in DATASET_COMPLETE:
            data[0] -= start_time
            csvwriter.writerow(data)            

if __name__ == '__main__':
    main()