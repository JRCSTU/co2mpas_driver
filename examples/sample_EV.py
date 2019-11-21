import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # file path without extension of the file
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar_cleaned'))

    car_id = 47844

    db = rno.load_db_to_dictionary(db_path)
    selected_car = rno.get_vehicle_from_db(db, car_id, electric=True)

    curves, start_stop = mf.get_ev_curve_main(selected_car)

    from co2mpas_driver.model import define_discrete_acceleration_curves as func
    discrete_acceleration_curves = func(curves, *start_stop)
    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'], 'x')
        plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    simple_run()
