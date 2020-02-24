import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    car_id = 47844

    # How to use co2mpas_driver library
    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(dict(vehicle_id=47844))['outputs']
    discrete_acceleration_curves = sol['discrete_acceleration_curves']
    discrete_deceleration_curves = sol['discrete_deceleration_curves']
    # start = sol['start']
    # stop = sol['stop']
    # driver.plot(1)

    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])
        plt.grid()

    for d in discrete_deceleration_curves:
        plt.plot(d['x'], d['y'])
        plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
