import os
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from new_MFC.common import curve_functions as mf
from new_MFC.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    db_name = '../db/EuroSegmentCar'
    car_id = 47844

    # file path without extension of the file
    db_name = os.path.dirname(db_name) + '/' + \
              os.path.splitext(os.path.basename(db_name))[0]

    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id, electric=True)

    Curves, StartStop = mf.get_ev_curve_main(selected_car)

    for gear, curve in enumerate(Curves):
        start = StartStop[0][gear]
        stop = min(StartStop[1][gear], 70)
        x = np.arange(start, stop, 1)
        y = curve(x)
        plt.plot(x, y, 'x')
        plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
