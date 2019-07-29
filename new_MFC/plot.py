import logging
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import schedula as sh

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(name='plot')


@sh.add_function(dsp)
def plot_simulation_result(times, speeds, acceleration, Start, Stop, Curves):
    """
    Plot simulation result.

    :param times:
        Sample time series.
    :type times: np.array

    :param speeds:
        Speed of the vehicle.
    :type speeds: list

    :param acceleration:
        Acceleration of vehicle.
    :type acceleration: list

    :param Start:
        Simulation start time.
    :type Start: list

    :param Stop:
        Simulation stop time.
    :type Stop: list

    :param Curves:
        Final acceleration curves.
    :type Curves: list
    :return:
    """
    plt.figure('Time-Speed')
    plt.plot(times, speeds[1:])
    plt.grid()
    plt.figure('Speed-Acceleration')
    plt.plot(speeds[1:], acceleration[1:])
    plt.grid()
    plt.figure('Acceleration-Time')
    plt.plot(times, acceleration[1:])
    plt.grid()

    plt.figure('Speed-Acceleration')
    for i, gear_curve in enumerate(Curves):
        sp_bins = np.arange(Start[i], Stop[i] + 0.1, 0.1)
        accelerations = gear_curve(sp_bins)
        plt.plot(sp_bins, accelerations, 'k')
    plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':
    output_plot_folder = osp.join(osp.dirname(__file__), 'output')