"""
crazyflie_pilot.py

This module provides the main control logic for piloting a Crazyflie drone, including flight path execution,
data logging, and measurement collection. It supports various predefined flight patterns (e.g., Snake, Cage, ...),
handles sensor data acquisition, and saves flight logs for later evaluation.

-----------------------------
Testing Parameters:
- LOGS_SAVE (bool): Enable or disable logging.
- MANUAL_TESTING_PARAMETERS (bool): Enable manual testing mode for parameter selection, if False input console is utilized.
- FLIGHTPATH (list): List of available flight paths for the drone.
- USE_FLIGHTPATH (int): Index of the flight path to use from the FLIGHTPATH list.
- FLIGHTHEIGHT (float): Height at which the drone will fly, in meters.
- DISTANCE (int): Distance between measurement points (identical to simulation), in decimeters.
- WINDOW_SIZE (list): Size of the flight window in decimeters.
- PADDING (int): Padding around the flight window in decimeters, added to ensure the drone flies in the middle of the grid.
- CRAZYFLIE_URI (list): Range of Crazyflie URIs to search for, specified as [start, end].

Constants:
- INITIAL_POSITION (x,y,height): Initial position of the drone before start.
- POSITION_ESTIMATE (x,y,height): Estimated position of the drone during flight.
- DATASET_FLIGHTPATH: List to store data during the flight, beginning from start to end of flight pattern.
- DATASET_COMPLETE: List to store complete dataset including all measurements.
- FILENAME: Base name for the CSV files where flight data will be saved.
- START_TIME (bool): Flag to indicate if the flight has started, used for logging purposes.

-----------------------------
Functions:
- main():
    Entry point for the script. Handles parameter selection and starts the measurement routine.

- crazyflie_take_measurements():
    Connects to the Crazyflie, sets up logging, and executes the selected flight pattern.

- fly_snake_pattern(), fly_cage_pattern(), fly_position_pattern(), fly_start_land(), fly_snake_pattern_absolute():
    Implement specific flight routines using the MotionCommander or crazyflie (for absolute positioning) interface.

- log_pos_callback():
    Callback for logging position and sensor data during flight, implemented with or without asynchronous logging.

- save_dataset_to_csv():
    Exports collected datasets to CSV files for further analysis.

- Utility functions:
    - set_starting_position(), fly_landing_position(), fly_take_off(), etc.

-----------------------------
Dependencies:
- cflib, threading, logging, csv, pathlib, concurrent.futures
- Custom modules: logs.logger, flightpaths, parameter_input_console

-----------------------------
Usage:
Run this script directly to collect sensor data with the drone, with:
python crazyflie_pilot.py

"""
import logs.logger as logger
import logging
import flightpaths
import parameter_input_console
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
from concurrent.futures import ThreadPoolExecutor


deck_attached_event = Event()
INITIAL_POSITION = [0, 0, 0]
POSITION_ESTIMATE = [0, 0, 0]
#GAS_DISTRIBUTION = [0, 0, 0] for testing purposes, not used in this script
DATASET_FLIGHTPATH = []
DATASET_COMPLETE = []
FILENAME="GSL"
START_TIME = False

TESTING_PARAMETERS = {
              "LOGS_SAVE": True,
              "MANUAL_TESTING_PARAMETERS": True,
              "FLIGHTPATH" : ["Nothing","StartLand","Snake","Snake_Long_Sampling","Snake_Absolute","Cage","TestPositioning"],
              "USE_FLIGHTPATH": 2,      # 0: Nothing, 1: StartLand, 2: Snake, 3: Snake_Long_Sampling, 4: Snake_Absolute, 5: Cage, 6: TestPositioning
              "FLIGHTHEIGHT": 0.95,     # in m
              "DISTANCE": 4,            # in dm
              'WINDOW_SIZE' : [20, 20], # in dm
              'PADDING': 1,             # in dm + 5 to fly in the middle of the 1dm x 1dm grid
              'CRAZYFLIE_URI': [0,5]    # start and end of the crazyflie number to search for
  }


def main():
    logger.logging_config(logs_save=TESTING_PARAMETERS['LOGS_SAVE'], filename="crazyflie_test_pilot")
    
    if TESTING_PARAMETERS["MANUAL_TESTING_PARAMETERS"]:        
        flightpath = TESTING_PARAMETERS["FLIGHTPATH"][TESTING_PARAMETERS["USE_FLIGHTPATH"]]
        flightheight = TESTING_PARAMETERS["FLIGHTHEIGHT"]
        distance = TESTING_PARAMETERS["DISTANCE"]
        URI = parameter_input_console.choose_model(start=TESTING_PARAMETERS['CRAZYFLIE_URI'][0], end=TESTING_PARAMETERS['CRAZYFLIE_URI'][1])
    else:  
        URI = parameter_input_console.choose_model()
        flightpath=parameter_input_console.choose_flightpath()
        flightheight=parameter_input_console.choose_flightheight()
        distance=parameter_input_console.choose_distance()
        URI = parameter_input_console.choose_model()
    crazyflie_take_measurements(URI=URI, flightpath=flightpath, flightheight=flightheight, distance=distance)
    save_dataset_to_csv()
    print("DONE")

def crazyflie_take_measurements(URI=uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703'), flightpath="Snake", flightheight=50, distance=5):
    logging.info("Crazyflie takes measurements.")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
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

        scf.cf.log.add_config(logconf)
        global FILENAME
        FILENAME=f"GSL_{flightpath}_{flightheight}"
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            logging.error('[EXIT] No flow deck detected!')
            print('[EXIT] No flow deck detected!')
            sys.exit(1)


        coordinates=flightpaths.flightpath_to_coordinates(flightpath=flightpath,window_shape=TESTING_PARAMETERS['WINDOW_SIZE'],pad=TESTING_PARAMETERS['PADDING'],distance=distance)
        logconf.start()    
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
            print(f"Flightpath: {flightpath}, Flightheight: {flightheight}, Distance: {distance}")                        
            if flightpath=="Snake":
                longer_sampling = False
            else:
                longer_sampling = True                
            fly_snake_pattern(scf, flightheight, coordinates, longer_sampling)
        elif(flightpath=="Snake_Absolute"):
            print(f"Flightpath: {flightpath}, Flightheight: {flightheight}, Distance: {distance}")                        
            fly_snake_pattern_absolute(scf, flightheight, coordinates)
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



def fly_snake_pattern(scf, flightheight, coordinates, longer_sampling=True):
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

    with MotionCommander(scf, default_height=flightheight) as mc:
        fly_take_off(mc,flightheight)      
        time.sleep(0.6) 
        starting_position=relative_positions.pop(0)
        mc.move_distance(starting_position[0],starting_position[1],0)
        START_TIME = True  
        time.sleep(0.6) 
        for koordinate in relative_positions:
            mc.move_distance(koordinate[0],koordinate[1],0) 
            if longer_sampling: 
                time.sleep(0.5)
            if(koordinate[1]):
                print("turn")
        START_TIME = False
        fly_landing_position(mc)  
        time.sleep(2)        
        mc.stop()
        time.sleep(2)   


def fly_snake_pattern_absolute(cf, flightheight, coordinates):
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
    for koordinate in relative_positions:
        for x in range(10):
            cf.commander.send_hover_setpoint(koordinate[0],koordinate[1], 0, flightheight)
            time.sleep(0.1)
        #time.sleep(0.2)
        if(koordinate[1]):
            print("turn")
    START_TIME = False
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
        START_TIME = False

        fly_landing_position(mc)    
        time.sleep(2)        
        mc.stop()
        time.sleep(2)        


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

    add_dataset_complete_async(timestamp, data)
    # without using the async function: 
    # DATASET_COMPLETE.append([timestamp, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'), data['sgp30.value1L'],data['sgp30.value1R']])

    if START_TIME:
        add_dataset_flightpath_async(timestamp, data)
        # without using the async function: 
        # DATASET_FLIGHTPATH.append([timestamp, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'), data['sgp30.value1L'],data['sgp30.value1R']])


executor = ThreadPoolExecutor(max_workers=1)
def add_dataset_complete_async(t, data):
    executor.submit(save_to_complete_list, t, data)

def save_to_complete_list(t, data):
    global DATASET_COMPLETE
    DATASET_COMPLETE.append([t, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'),data['sgp30.value1L'],data['sgp30.value1R']])

def add_dataset_flightpath_async(t, data):
    executor.submit(save_to_flight_list, t, data)

def save_to_flight_list(t, data):
    global DATASET_FLIGHTPATH
    DATASET_FLIGHTPATH.append([t, format(data['stateEstimate.x'], '.10f'), format(data['stateEstimate.y'], '.10f'),format(data['stateEstimate.z'], '.10f'),data['sgp30.value1L'],data['sgp30.value1R']])





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