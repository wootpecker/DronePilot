import cflib.crtp
import logs.logger as logger
import logging
import flightpaths
import parameters

import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.utils import uri_helper
import csv
from pathlib import Path

LOGS_SAVE = True
#URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage","TestPositioning"]
deck_attached_event = Event()
INITIAL_POSITION = [0, 0, 0]
POSITION_ESTIMATE = [0, 0, 0]
GAS_DISTRIBUTION = [0, 0, 0]
FILENAME="GSL"
START_TIME = None
FILEPATH="data/test.csv"

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_pilot")
    URI=parameters.choose_model()
    flightpath=parameters.choose_flightpath()
    flightheight=parameters.choose_flightheight()
    distance=parameters.choose_distance()
    crazyflie_take_measurements(URI=URI, flightpath=flightpath, flightheight=flightheight, distance=distance)
    

def crazyflie_take_measurements(URI=uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703'), flightpath="Snake", flightheight=50, distance=5):
    logging.info("Crazyflie takes measurements.")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        #scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
        #                                 cb=param_deck_flow)
        #time.sleep(2)

        scf.cf.param.add_update_callback(group='deck', name=f'bcFlow2',
                                             cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('range.zrange', 'uint16_t')
        logconf.add_variable('sgp30.value1L', 'uint16_t')   #configuration_data.py
        logconf.add_variable('sgp30.value1R', 'uint16_t')        
        #set_initial_position(scf, 0, 0, 0)

        scf.cf.log.add_config(logconf)
        global START_TIME,FILENAME
        FILENAME=f"GSL_{flightpath}_{flightheight}"
        #START_TIME = time.time()*1000
        create_csv()
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            logging.error('[EXIT] No flow deck detected!')
            print('[EXIT] No flow deck detected!')
            #sys.exit(1)

        scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(1)
        scf.cf.param.set_value('kalman.resetEstimation', '0') 
        time.sleep(1)
        window_shape=init_windowshape(logconf)
        coordinates=flightpaths.flightpath_to_coordinates(flightpath=flightpath,window_shape=[15,15],distance=6)
        logconf.start()    
        time.sleep(1)
        scf=set_initial_position(scf,flightheight)
        #time.sleep(2)
        if(flightpath=="Nothing"):
            print("Flightpath: Nothing")
            time.sleep(20)
            logconf.stop()
            return
        if(flightpath=="StartLand"):
            print("Flightpath: StartLand")            
            fly_start_land(scf, flightheight)
        if(flightpath=="TestPositioning"):
            print("Flightpath: TestPositioning")            
            fly_position_pattern(scf,flightheight)

        if(flightpath=="Snake"):
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

    with MotionCommander(scf, default_height=flightheight) as mc:
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
    relative_positions=[]
    coordinates.insert(0,[0,0])
    for x in range(len(coordinates)-1):
        relative_positions.append([coordinates[x+1][0]-coordinates[x][0],coordinates[x+1][1]-coordinates[x][1]])
    print(relative_positions)
    with MotionCommander(scf, default_height=flightheight) as mc:
        print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")        
        time.sleep(5)        
        for koordinate in relative_positions:
            mc.move_distance(koordinate[0],koordinate[1],0) 
            time.sleep(0.1)
            if(koordinate[1]):
                print("turn")
       # print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        #print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        fly_landing_position(mc)  
        time.sleep(2)        
        mc.stop()
        time.sleep(2)   

             


def fly_cage_pattern(scf, flightheight, coordinates):
    relative_positions=[]
    coordinates.insert(0,[0,0])
    for x in range(len(coordinates)-1):
        relative_positions.append([coordinates[x+1][0]-coordinates[x][0],coordinates[x+1][1]-coordinates[x][1]])
    print(relative_positions)
    with MotionCommander(scf, default_height=flightheight) as mc:
        time.sleep(3)        
        for koordinate in relative_positions:
            mc.move_distance(koordinate[0],koordinate[1],0)
            time.sleep(0.1)

        fly_landing_position(mc)    
       # difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
       # mc.move_distance(difference[0],difference[1],0)
        time.sleep(2)        
        mc.stop()
        time.sleep(2)        



    #for koordinate in coordinates:

def fly_landing_position(mc):
    print("landing")
    difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
    print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
    print(f"INITIAL_POSITION: {INITIAL_POSITION}")
    print(f"difference: {difference}")     
    while(abs(difference[0])>0.1 or abs(difference[1])>0.1):
        
        difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
        mc.move_distance(difference[0],difference[1],0)    
        print(f"POSITION_ESTIMATE: {POSITION_ESTIMATE}")
        print(f"INITIAL_POSITION: {INITIAL_POSITION}")
        print(f"difference: {difference}")        
        time.sleep(2)


def fly_start_land(scf, flightheight):
    with MotionCommander(scf, default_height=flightheight) as mc:
        print("Take off START")
        time.sleep(5)
        print("Take off Hover")
        
        mc.stop()
        time.sleep(5)
        print("Take off Land")



def fly_landing(scf, flightheight):
    with MotionCommander(scf, default_height=flightheight) as mc:
        print("Landing START")
        time.sleep(3)
        print("Landing END")        
        mc.stop()
        time.sleep(3)
        print("mc.stop")
        time.sleep(5)



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
    #print(data)
    global POSITION_ESTIMATE, START_TIME, GAS_DISTRIBUTION, INITIAL_POSITION
    if START_TIME==None:
        START_TIME=timestamp
        INITIAL_POSITION=[data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']]
    #t = int(time.time() * 1000)
    POSITION_ESTIMATE[0] = data['stateEstimate.x']
    POSITION_ESTIMATE[1] = data['stateEstimate.y']
    POSITION_ESTIMATE[2] = data['stateEstimate.z']
    GAS_DISTRIBUTION[0] = data['range.zrange']
    GAS_DISTRIBUTION[1] = data['sgp30.value1L']
    GAS_DISTRIBUTION[2] = data['sgp30.value1R']
    save_to_csv(timestamp-START_TIME,POSITION_ESTIMATE,GAS_DISTRIBUTION)

    #with open('/home/hujiao/Desktop/0.15-0.5_logging.tsv', 'a+', newline='') as f:
       # tsv_w = csv.writer(f, delimiter='\t')
        #tsv_w.writerow([int(t), format(POSITION_ESTIMATE[0], '.10f'), format(POSITION_ESTIMATE[1], '.10f'), format(POSITION_ESTIMATE[2], '.10f')])

def set_initial_position(scf,flightheight):
    scf.cf.param.set_value('kalman.initialX', INITIAL_POSITION[0])
    scf.cf.param.set_value('kalman.initialY', INITIAL_POSITION[1])
    scf.cf.param.set_value('kalman.initialZ', flightheight)
    return scf

def init_windowshape(logconf):
    #logconf.start()
    #global POSITION_ESTIMATE
    #end_x,end_y=POSITION_ESTIMATE[0],POSITION_ESTIMATE[1]
    #logconf.stop()
    pass
    


def create_csv():
    global FILEPATH
    target_dir_path = Path("data")
    target_dir_path.mkdir(parents=True, exist_ok=True)
    file_path = target_dir_path / f"{FILENAME}.csv"
    FILEPATH=file_path
    file_path.touch(exist_ok=True)
    with open(file_path, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter=',')
        tsv_w.writerow(['Time', 'X', 'Y', 'Z', 'DiffX', 'DiffY', 'DiffZ', 'Zrange', 'Gas1L', 'Gas1R'])



def save_to_csv(t,position_estimate,gas_distribution):
    with open(FILEPATH, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([t,format(position_estimate[0], '.10f'), format(position_estimate[1], '.10f'), format(position_estimate[2], '.10f'),    format(position_estimate[0]-INITIAL_POSITION[0], '.10f'), format(position_estimate[1]-INITIAL_POSITION[1], '.10f'), format(position_estimate[2]-INITIAL_POSITION[2], '.10f')      , gas_distribution[0], gas_distribution[1], gas_distribution[2]])

def save_to_csv2(t,position_estimate):
    with open('data/test.tsv', 'a+', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        #print(int(t), format(position_estimate[0], '.10f'), format(position_estimate[1], '.10f'), format(position_estimate[2], '.10f'))
        tsv_w.writerow([int(t), format(position_estimate[0], '.10f'), format(position_estimate[1], '.10f'), format(position_estimate[2], '.10f')])


if __name__ == '__main__':
    main()