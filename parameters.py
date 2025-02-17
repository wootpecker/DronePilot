import crazyflie_init
import sys
import logging
from cflib.utils import uri_helper

FLIGHT_PATH = ["Nothing","StartLand","Snake", "Cage"]



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
    return FLIGHT_PATH[2]
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



def choose_distance():
    return 5
    output=""
    distance=input(f"Choose a distance between measurements in dm(=10cm) (1-20): ")
    try:
        distance=int(distance)
    except:
        distance=-1
    while(int(distance)<10 or int(distance)>100):
        if distance==-1:
            distance=input(f"Wrong Input: No Number.\nChoose a distance between measurements in dm(=10cm) (1-20): ")
        else:    
            distance=input(f"Wrong Input: {distance}.\nChoose a distance between measurements in dm(=10cm) (1-20): ")
        try:
            distance=int(distance)
        except:
            distance=-1
    logging.info(f"Crazyflie Pilot choose distance: {distance}.")            
    return distance