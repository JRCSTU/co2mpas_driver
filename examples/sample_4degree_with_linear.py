import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver.load import dsp as load
from co2mpas_driver.model import dsp as model
from co2mpas_driver import dsp as core
my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # Vehicles database path
    db_path = 'C:/Apps/new_MFC/co2mpas_driver/db/EuroSegmentCar.csv'
    # db_path = osp.join(my_dir, '../db', 'EuroSegmentCar.csv')
    input_path = 'C:/Apps/new_MFC/co2mpas_driver/template/sample.xlsx'
    # user inputs
    inputs = {  # Category item
        'inputs': {'gear_shifting_style': 0.7},  # gear shifting can take value from 0(timid driver)
        # to 1(aggressive driver)
        'vehicle_inputs': {'vehicle_mass': 0.4},
        'time_series': {'times': list(range(2, 23))}
    }
    selected_car = core(dict(db_path=db_path, input_path=input_path, inputs=inputs)).plot()
    # dsp.register(memo={}).plot()

    output = model(selected_car, outputs=['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                                          'discrete_acceleration_curves'])

    for d in output['discrete_acceleration_curves']:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
