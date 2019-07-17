import os
from os import path as osp
import matplotlib.pyplot as plt
from new_MFC.common import functions as fun
from new_MFC.common import curve_functions as mf
from new_MFC.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # file path without extension of the file
    db_name = '../db/EuroSegmentCar'
    car_id = 47844

    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id, electric=True)

    curves, start_stop = mf.get_ev_curve_main(selected_car)

    for gear, curve in enumerate(curves):
        x, y = fun.calculate_curve_coordinates(curve, gear, *start_stop)
        plt.plot(x, y, 'x')
        plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
