import cflib.crtp
import logging



def initialize():
    logging.info('[INIT] Initialize Crazyflie drivers.')
    cflib.crtp.init_drivers()
    #print('Scanning interfaces for Crazyflies...')
    logging.info('[INIT] Scanning interfaces for Crazyflies...')
    #available = cflib.crtp.scan_interfaces(address=int('E7E7E7E7E7', 16))
    #end=int('E7',16)
    end = 10
    possible_crazyflies = []
    for i in range(end):
        address=int('E7E7E7E700', 16)+i
        available = cflib.crtp.scan_interfaces(address=address)
        if len(available) > 0:
            possible_crazyflies.append(address)
            for i in available:
                logging.info(f'Crazyflies found: {i[0]}')
    return possible_crazyflies            

        