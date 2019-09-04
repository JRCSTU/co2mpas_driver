import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # Vehicles database path
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar'))
    car_id = 39393
    gs_style = 0.8  # gear shifting can take value from 0(timid driver)
    degree = 2

    db = rno.load_db_to_dictionary(db_path)
    selected_car = rno.get_vehicle_from_db(db, car_id)

    curves, cs_acc_per_gear, start_stop, gs = mf.gear_curves_n_gs_from_poly(
        selected_car, gs_style, degree)

    from co2mpas_driver.model import define_discrete_acceleration_curves as func
    discrete_acceleration_curves = func(curves, *start_stop)
    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
