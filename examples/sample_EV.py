import numpy as np
import matplotlib.pyplot as plt
import curve_functions as mf
import reading_n_organizing as rno
import vehicle_functions as vf

def simple_run():
    db_name = '../db/EuroSegmentCar'
    car_id = 47844


    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id, electric = True)

    Curves,StartStop = mf.get_ev_curve_main(selected_car)

    for gear, curve in enumerate(Curves):
        start = StartStop[0][gear]
        stop = 50#min(StartStop[1][gear], 70)
        x = np.arange(start, stop, 1)
        y = curve(x)
        plt.plot(x, y,'x')
        plt.grid()
    plt.show()

    return 0


simple_run()
