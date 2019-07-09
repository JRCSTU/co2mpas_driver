import os
import numpy as np
import matplotlib.pyplot as plt
from new_MFC.common import curve_functions as mf
from new_MFC.common import reading_n_organizing as rno


def simple_run(db_name):
    # Database of vehicles with a unique id
    # db_name = '../db/EuroSegmentCar'
    car_id = 35135  # Category item
    gs_style = 0.8  # gear shifting can take value from 0(timid driver)
    # to 1(aggressive driver)

    # file path without extension of the file
    db_name = os.path.dirname(db_name) + '/' + \
        os.path.splitext(os.path.basename(db_name))[0]

    # reading csv file into a dictionary
    db = rno.load_db_to_dictionary(db_name)
    # Select a car based on its id
    selected_car = rno.get_vehicle_from_db(db, car_id)

    curves, cs_acc_per_gear, start_stop, gs = \
        mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)

    for gear, curve in enumerate(curves):
        start = start_stop[0][gear]
        stop = start_stop[1][gear]
        x = np.arange(start, stop, 0.2)
        y = curve(x)
        plt.plot(x, y)
    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
