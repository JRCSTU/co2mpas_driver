import os
from os import path as osp
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as core
import schedula as sh
my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    # Vehicles database path
    db_path = 'C:/Apps/new_MFC/co2mpas_driver/db/EuroSegmentCar.csv'
    # db_path = osp.join(my_dir, '../db', 'EuroSegmentCar.csv')
    input_path = 'C:/Apps/new_MFC/co2mpas_driver/template/sample.xlsx'
    # user inputs
    inputs = {
        'vehicle_id': 35135,  # A sample car id from the database
        'inputs': {'gear_shifting_style': 0.7},  # gear shifting can take value
        # from 0(timid driver) to 1(aggressive driver)
        'time_series': {'times': list(range(2, 23))}
    }

    output = sh.selector(['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                          'discrete_acceleration_curves'],
                sh.selector(['outputs'], sh.selector(['outputs'],
                      core(dict(db_path=db_path, input_path=input_path,
                                inputs=inputs), outputs=['outputs'], shrink=True)))['outputs'])

    Curves, poly_spline, Start, Stop, gs, discrete_acceleration_curves = \
        output['Curves'], output['poly_spline'], output['Start'], \
        output['Stop'], output['gs'], output['discrete_acceleration_curves']

    # dsp.register(memo={}).plot()

    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
