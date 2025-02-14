import cflib.crtp
import logs.logger as logger
import logging
import crazyflie_init

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
FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage"]
deck_attached_event = Event()
INITIAL_POSITION = [0, 0, 0]
POSITION_ESTIMATE = [0, 0, 0]
GAS_DISTRIBUTION = [0, 0, 0]
FILENAME="GSL"
START_TIME = None
FILEPATH="data/test.csv"

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_pilot")
    URI=choose_model()
    flightpath=choose_flightpath()
    flightheight=choose_flightheight()
    crazyflie_take_measurements(URI=URI, flightpath=flightpath, flightheight=flightheight)
    

def crazyflie_take_measurements(URI=uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703'), flightpath="Snake", flightheight=50):
    logging.info("Crazyflie takes measurements.")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
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
            sys.exit(1)

        scf.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(1)
        scf.cf.param.set_value('kalman.resetEstimation', '0') 
        time.sleep(3)

        logconf.start()    
        time.sleep(1)
        scf=set_initial_position(scf,flightheight)
        #time.sleep(2)
        if(flightpath=="Nothing"):
            print("Flightpath: Nothing")
            time.sleep(10)
            logconf.stop()
            return
        if(flightpath=="StartLand"):
            print("Flightpath: StartLand")            
            fly_start_land(scf, flightheight)
        if(flightpath=="Snake"):
            print("Flightpath: Snake")                        
            fly_snake_pattern(scf, flightheight)
        elif(flightpath=="Cage"):
            print("Flightpath: Cage")                        
            fly_cage_pattern(scf, flightheight)
        #take_off_simple(scf)
        # move_linear_simple(scf)
        # move_box_limit(scf)
        logconf.stop()



def fly_snake_pattern(scf, flightheight):
    with MotionCommander(scf, default_height=flightheight) as mc:
        time.sleep(2)
        print("0.3")
        mc.move_distance(0.3,0.3,0)
        time.sleep(2)
        print("0.6")
        mc.move_distance(0.6,0,0)
        time.sleep(2)
        print("0")
        mc.move_distance(-0.3,-0.6,0)
        print(POSITION_ESTIMATE)
        print(INITIAL_POSITION)
        difference = [INITIAL_POSITION[i] - POSITION_ESTIMATE[i]  for i in range(len(POSITION_ESTIMATE))]
        print(difference)
        mc.move_distance(difference[0],difference[1],0)
        time.sleep(2)


def fly_cage_pattern(scf, flightheight):
    pass

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




def choose_model():
    crazyflies=crazyflie_init.initialize()
    #crazyflies=['E7E7E7E703','E7E7E7E7E2','E7E7E7E7E7']
    if len(crazyflies)==0:
        logging.info("[EXIT] No Crazyflies found.")
        print("[EXIT] No Crazyflies found.")
        sys.exit(1)
    elif len(crazyflies)==1:
        logging.info(f"One Crazyflie found: {crazyflies}.")
    else:
        output=""
        for cf in range(len(crazyflies)):
            output=output+f"{cf} )  Crazyflie found: {crazyflies[cf]} \n"
        crazyflie=input(f"Multiple Crazyflies found: \n{output}Choose a Crazyflie to connect to (0-{len(crazyflies)-1}): ")
        try:
            crazyflie=int(crazyflie)
        except:
            crazyflie=-1
        while(int(crazyflie)<0 or int(crazyflie)>=len(crazyflies)):
            if crazyflie==-1:
                crazyflie=input(f"Wrong Input: No Number.\nChoose a Crazyflie to connect to (0-{len(crazyflies)-1}): ")
            else:    
                crazyflie=input(f"Wrong Input: {crazyflie}.\nChoose a Crazyflie to connect to (0-{len(crazyflies)-1}): ")
            try:
                crazyflie=int(crazyflie)
            except:
                crazyflie=-1

        crazyflies=[crazyflies[int(crazyflie)]]
    logging.info(f"Crazyflie Pilot Started with {crazyflies[0]}.")
    uri=uri_helper.uri_from_env(default=f'radio://0/80/2M/{crazyflies[0]}')
    return uri

def choose_flightpath():
    #return FLIGHT_PATH[0]
    output=""
    for fp in range(len(FLIGHT_PATH)):
        output=output+f"{fp} )  Flightpath: {FLIGHT_PATH[fp]} \n"
    flightpath=input(f"{output}Choose a Flightpath (0-{len(FLIGHT_PATH)-1}): ")
    try:
        flightpath=int(flightpath)
    except:
        flightpath=-1
    while(int(flightpath)<0 or int(flightpath)>=len(FLIGHT_PATH)):
        if flightpath==-1:
            flightpath=input(f"Wrong Input: No Number.\nChoose a Flightpath (0-{len(FLIGHT_PATH)-1}): ")
        else:    
            flightpath=input(f"Wrong Input: {flightpath}.\nChoose a Flightpath (0-{len(FLIGHT_PATH)-1}): ")
        try:
            flightpath=int(flightpath)
        except:
            flightpath=-1
    logging.info(f"Crazyflie Pilot choose Flightpath: {FLIGHT_PATH[flightpath]}.")            
    return FLIGHT_PATH[flightpath]


def choose_flightheight():
    return 0.3
    output=""
    flightheight=input(f"Choose a Height for the Experiment in cm (10-100): ")
    try:
        flightheight=int(flightheight)
    except:
        flightheight=-1
    while(int(flightheight)<10 or int(flightheight)>100):
        if flightheight==-1:
            flightheight=input(f"Wrong Input: No Number.\nChoose a Height for the Experiment in cm (10-100): ")
        else:    
            flightheight=input(f"Wrong Input: {flightheight}.\nChoose a Height for the Experiment in cm (10-100): ")
        try:
            flightheight=int(flightheight)
        except:
            flightheight=-1
    logging.info(f"Crazyflie Pilot choose Flightheight: {flightheight}.")            
    return flightheight


def param_deck_flow(_, value_str):
    value = int(value_str)
    logging.info(f"Deck Flow Value: {value}")
    if value:
        deck_attached_event.set()
        logging.info(f"Flow Deck is attached!")        
    else:
        logging.info(f"[EXIT] No Flow Deck detected or attached!")
        print(f"[EXIT] No Flow Deck detected or attached!")
        
        sys.exit(1)


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