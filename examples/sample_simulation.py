import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver import dsp
import schedula as sh

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)


def simple_run():
    """:parameters of the simulation"""
    # Vehicle databased based on the Euro Car Segment classification
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar.csv'))

    # input file path
    input_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                      'co2mpas_driver', 'template',
                                      'sample.xlsx'))

    # The simulation step in seconds
    sim_step = 0.1

    # Duration of the simulation in seconds.
    duration = 100

    # sample time series
    times = np.arange(0, duration + sim_step, sim_step)

    inputs = {
        'vehicle_id': 35135,  # A sample car id from the database
        'inputs': {'gear_shifting_style': 0.7, 'starting_speed': 0,
                   'desired_velocity': 40,
                   'driver_style': 1},  # gear shifting can take value
        # from 0(timid driver) to 1(aggressive driver)
        'time_series': {'times': times}
    }

    core = dsp(dict(db_path=db_path, input_path=input_path, inputs=inputs),
               outputs=['outputs'], shrink=True)

    # plots simulation model
    core.plot()

    # outputs of the dispatcher
    outputs = sh.selector(['outputs'], sh.selector(['outputs'], core))

    # select the desired output
    output = sh.selector(['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                          'discrete_acceleration_curves', 'velocities',
                          'accelerations', 'transmission'], outputs['outputs'])

    """
    The final acceleration curvers (Curves), the engine acceleration potential 
    curves (cs_acc_per_gear), before the calculation of the resistances and the
    limitation due to max possible acceleration (friction) .
    """
    curves, poly_spline, start, stop, gs, discrete_acceleration_curves, \
    velocities, accelerations, transmission, discrete_acceleration_curves = \
        output['Curves'], output['poly_spline'], \
        output['Start'], output['Stop'], output['gs'], \
        output['discrete_acceleration_curves'], output['velocities'], \
        output['accelerations'], output['transmission'], \
        output['discrete_acceleration_curves']

    # ******************* Plot*************************
    """Plot"""
    plt.figure('Time-Speed')
    plt.plot(times, velocities)
    plt.grid()
    plt.figure('Speed-Acceleration')
    plt.plot(velocities, accelerations)
    plt.grid()
    plt.figure('Acceleration-Time')
    plt.plot(times, accelerations)
    plt.grid()

    plt.figure('Speed-Acceleration')
    for curve in discrete_acceleration_curves:
        sp_bins = list(curve['x'])
        acceleration = list(curve['y'])
        plt.plot(sp_bins, acceleration, 'k')
    plt.show()
    return 0


if __name__ == '__main__':
    simple_run()
