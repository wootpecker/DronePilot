import cflib.crtp
import logging



def initialize():
    logging.info('[INIT] Initialize Crazyflie drivers.')
    cflib.crtp.init_drivers()
    logging.info('[INIT] Scanning interfaces for Crazyflies...')
    end = 5
    possible_crazyflies = []
    for i in range(end):
        address=int('E7E7E7E700', 16)+i
        address_str=f'E7E7E7E7{i:02d}'
        available = cflib.crtp.scan_interfaces(address=address)
        if len(available) > 0:
            possible_crazyflies.append(address_str)
            for i in available:
                logging.info(f'Crazyflies found: {i[0]}')
    return possible_crazyflies            

        