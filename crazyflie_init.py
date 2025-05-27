"""
crazyflie_init.py

This module initializes the Crazyflie drivers and scans for available Crazyflie drones on the network.

-----------------------------
Functions:
- initialize(start, end):
    Initializes the Crazyflie drivers and scans for available Crazyflie interfaces within the specified address range.
    Returns a list of found Crazyflie addresses.

-----------------------------
Dependencies:
- cflib.crtp, logging

-----------------------------
Usage:
- Import and call initialize() to set up drivers and scan for Crazyflies
"""

import cflib.crtp
import logging



def initialize(start=0, end=5):
    logging.info('[INIT] Initialize Crazyflie drivers.')
    cflib.crtp.init_drivers()
    logging.info('[INIT] Scanning interfaces for Crazyflies...')
    possible_crazyflies = []
    for i in range(start, end):
        address=int('E7E7E7E700', 16)+i
        address_str=f'E7E7E7E7{i:02d}'
        available = cflib.crtp.scan_interfaces(address=address)
        if len(available) > 0:
            possible_crazyflies.append(address_str)
            for i in available:
                logging.info(f'Crazyflies found: {i[0]}')
    return possible_crazyflies            

        