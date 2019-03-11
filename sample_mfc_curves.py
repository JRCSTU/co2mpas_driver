import numpy as np
import matplotlib.pyplot as plt
import main_functions as mf
import reading_n_organizing as rno


def simple_run(speed, rpm):
    db_name = 'car_db_sample'
    car_id = 24211
    gs_style = 0.8
    degree = 2

    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id)

    Curves, cs_acc_per_gear, StartStop, gs = mf.gear_curves_n_gs(selected_car,gs_style,degree)

    for gear, curve in enumerate(Curves):
        start = StartStop[0][gear]
        stop = min(StartStop[1][gear], 50)
        x = np.arange(start, stop, 1)
        y = curve(x)
        plt.plot(x, y)
    plt.show()

    return 0


simple_run(speed=190 / 3.6, rpm=4000)
