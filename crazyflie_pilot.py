


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
from cflib.utils import uri_helper


LOGS_SAVE = False
#URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
FLIGHT_PATH = ["Snake", "Cage"]
deck_attached_event = Event()
position_estimate = [0, 0, 0]

def main():
    logger.logging_config(logs_save=LOGS_SAVE, filename="crazyflie_pilot")
    URI=choose_model()
    if URI==None:
        return
    flightpath=choose_flightpath()
    flightheight=choose_flightheight()
    crazyflie_take_measurements(URI=URI, flightpath=flightpath, flightheight=flightheight)
    pass

def crazyflie_take_measurements(URI=uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703'), flightpath="Snake", flightheight=50):
    logging.info("Crazyflie takes measurments.")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        logconf = LogConfig(name='Position', period_in_ms=10)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')

        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)





def choose_model():
    crazyflies=crazyflie_init.initialize()
    crazyflies=['E7E7E7E7E1','E7E7E7E7E2','E7E7E7E7E7']
    if len(crazyflies)==0:
        logging.info("[EXIT] No Crazyflies found.")
        return
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
        logging.info(f"Deck is attached!")        
    else:
        logging.info(f"Deck is NOT attached!")


def log_pos_callback(timestamp, data, logconf):
    print(data)
    global position_estimate
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']
    position_estimate[2] = data['stateEstimate.z']




if __name__ == '__main__':
    main()