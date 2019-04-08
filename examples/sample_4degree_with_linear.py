import numpy as np
import matplotlib.pyplot as plt
import curve_functions as mf
import reading_n_organizing as rno

'''
Example of how to extract the acceleration curves of a vehicle and the corresponding plot.
'''

def simple_run():
    # db_name = '../db/car_db_sample'
    db_name = '../db/EuroSegmentCar'

    # A sample car id from the database
    car_id = 39393

    # The gear shifting style as described in the TRR paper.
    gs_style = 0.8

    '''import vehicle object, curves and gear shifting strategy'''
    db = rno.load_db_to_dictionary(db_name)

    # The vehicle specs as returned from the database
    selected_car = rno.get_vehicle_from_db(db, car_id)

    '''
        The final acceleration curvers (Curves), the engine acceleration potential curves (cs_acc_per_gear),
        before the calculation of the resistances and the limitation due to max possible acceleration (friction) .
        '''
    Curves, cs_acc_per_gear, StartStop, gs = mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)

    for gear, curve in enumerate(Curves):
        start = StartStop[0][gear]
        stop = StartStop[1][gear]
        x = np.arange(start, stop, 0.2)
        y = curve(x)
        plt.plot(x, y)
    plt.show()

    return 0


simple_run()
