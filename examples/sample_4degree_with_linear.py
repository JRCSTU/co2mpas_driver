import numpy as np
import matplotlib.pyplot as plt
import curve_functions as mf
import reading_n_organizing as rno


def simple_run():
    db_name = '../db/EuroSegmentCar'
    car_id = 8183
    gs_style = 0.8
    degree = 4

    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id)

    Curves, cs_acc_per_gear, StartStop, gs = mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)

    for gear, curve in enumerate(Curves):
        start = StartStop[0][gear]
        stop = StartStop[1][gear]
        x = np.arange(start, stop, 1)
        y = curve(x)
        plt.plot(x, y,'x')
    plt.show()

    return 0


simple_run()
