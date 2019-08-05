import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # Database of vehicles with a unique id
    db_name = '../db/EuroSegmentCar'  # file path without extension of the file
    car_id = 35135  # Category item
    gs_style = 0.7  # gear shifting can take value from 0(timid driver)
    # to 1(aggressive driver)

    # reading csv file into a dictionary
    db = rno.load_db_to_dictionary(db_name)
    # Select a car based on its id
    selected_car = rno.get_vehicle_from_db(db, car_id)

    curves, cs_acc_per_gear, start_stop, gs = \
        mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)
    from co2mpas_driver.model import define_discrete_acceleration_curves as func
    discrete_acceleration_curves = func(curves, *start_stop)
    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()