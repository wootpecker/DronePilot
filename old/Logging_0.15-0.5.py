import logging
import sys
import time
import csv
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E703')

DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

U_shaped_Daten = [0, 0, 0, 0, 0, 0]


def move_linear_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)
        mc.turn_left(180)
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)


def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()



def log_pos_callback(timestamp, data, logconf):
    t = time.time()
    print(int(t))
    print(data)
    U_shaped_Daten[0] = data['stateEstimate.x']
    U_shaped_Daten[1] = data['stateEstimate.y']
    U_shaped_Daten[2] = data['stateEstimate.z']
    U_shaped_Daten[3] = data['range.zrange']
    U_shaped_Daten[4] = data['sgp30.value1L']
    U_shaped_Daten[5] = data['sgp30.value1R']
    with open('/home/hujiao/Desktop/0.15-0.5_logging.tsv', 'a+',newline='') as f:
         tsv_w = csv.writer(f, delimiter='\t')
         tsv_w.writerow([int(t), format(U_shaped_Daten[0],'.10f'), format(U_shaped_Daten[1],'.10f'), format(U_shaped_Daten[2],'.10f'), U_shaped_Daten[3], U_shaped_Daten[4], U_shaped_Daten[5]])
         #datum_row=[int(t),U_shaped_Daten[0], U_shaped_Daten[1], U_shaped_Daten[2], U_shaped_Daten[3], U_shaped_Daten[4], U_shaped_Daten[5]]
         #tsv_w.writerow(datum_row)


def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')


if __name__ == '__main__':
    cflib.crtp.init_drivers()

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
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        logconf.start()

        cf = scf.cf

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        for y in range(30):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 60)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
            time.sleep(0.1)

        for y in range(21):
            cf.commander.send_hover_setpoint(0,0,0,(30-y)/60)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0,0,0,0.15)
            time.sleep(0.1)

        for _ in range(100):
            cf.commander.send_hover_setpoint(0.2, 0, 0, 0.15)  # xv/(dm/s)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.15)
            time.sleep(0.1)

        for y in range(21):
            cf.commander.send_hover_setpoint(0,0,0,(y+9)/60)
            time.sleep(0.1)


        for _ in range(20):
            cf.commander.send_hover_setpoint(0,0,0,0.5)
            time.sleep(0.1)

        for _ in range(50):
            cf.commander.send_hover_setpoint(0, -0.2, 0, 0.5)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
            time.sleep(0.1)

        for y in range(21):
            cf.commander.send_hover_setpoint(0,0,0,(30-y)/60)
            time.sleep(0.1)

        for _ in range(100):
            cf.commander.send_hover_setpoint(-0.2, 0, 0, 0.15)
            time.sleep(0.1)

        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.15)
            time.sleep(0.1)

        for y in range(9):
            cf.commander.send_hover_setpoint(0,0,0,(9-y)/60)
            time.sleep(0.1)


        # for _ in range(50):
        #     cf.commander.send_hover_setpoint(0, 0.2, 0, 0.5)
        #     time.sleep(0.1)
        #
        # for _ in range(20):
        #     cf.commander.send_hover_setpoint(0, 0, 0, 0.5)
        #     time.sleep(0.1)

        # for y in range(30):
        #     cf.commander.send_hover_setpoint(0, 0, 0, (30 - y) / 60)
        #     time.sleep(0.1)

        cf.commander.send_stop_setpoint()

        logconf.stop()

