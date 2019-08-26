from os import chdir, path as osp
from co2mpas_driver.model import simulation as sp
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.common import gear_functions as fg
from co2mpas_driver import dsp
import schedula as sh

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """:parameters of the simulation"""
    inputs = {
        'vehicle_id': 35135,  # A sample car id from the database
        'inputs': {'gear_shifting_style': 0.7, 'starting_speed': 0,
                   'driver_style': 1},  # gear shifting can take value
        # from 0(timid driver) to 1(aggressive driver)
        'time_series': {'times': np.arange(21)}
    }

    # file path without extension of the file
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar.csv'))

    # input file path
    input_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                      'co2mpas_driver', 'template',
                                      'sample.xlsx'))

    core = dsp(dict(db_path=db_path, input_path=input_path, inputs=inputs),
               outputs=['outputs'], shrink=True)

    # plots simulation model
    core.plot()

    # ***********************************************************************
    """
    The final acceleration curvers (Curves), the engine acceleration potential 
    curves (cs_acc_per_gear), before the calculation of the resistances and the
    limitation due to max possible acceleration (friction).
    """
    # outputs of the dispatcher
    outputs = sh.selector(['outputs'], sh.selector(['outputs'], core))

    # select the desired output
    output = sh.selector(['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                          'discrete_acceleration_curves', 'velocities',
                          'accelerations'], outputs['outputs'])

    curves, poly_spline, start, stop, gs, discrete_acceleration_curves, \
    velocities, accelerations = output['Curves'], output['poly_spline'], \
                                output['Start'], output['Stop'], output['gs'], \
                                output['discrete_acceleration_curves'], \
                                output['velocities'], output['accelerations']
    return 0


if __name__ == '__main__':
    simple_run()
