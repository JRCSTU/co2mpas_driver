import logging
import numpy as np
from os import path as osp, mkdir
import matplotlib.pyplot as plt
import schedula as sh

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(name='plot')


@sh.add_function(dsp)
def plot_and_save_simulation_result(starting_velocity, vehicle_id,
                                    gear_shifting_style, driver_style, times,
                                    velocities, accelerations, positions, start,
                                    stop, curves):
    """
    Plot simulation result.

    :param starting_velocity:
        Current velocity.
    :type starting_velocity: float (m/s)

    :param vehicle_id:
        Vehicle Id.
    :type vehicle_id: int

    :param gear_shifting_style:
        Gear shifting style.
    :type gear_shifting_style: float

    :param driver_style:
        Driver style.
    :type driver_style: float

    :param times:
        Sample time series.
    :type times: np.array

    :param velocities:
        Speed of the vehicle.
    :type velocities: list

    :param accelerations:
        Acceleration of vehicle.
    :type accelerations: list

    :param positions:
        Position of vehicle.
    :type positions: list

    :param start:
        Simulation start time.
    :type start: list

    :param stop:
        Simulation stop time.
    :type stop: list

    :param curves:
        Final acceleration curves.
    :type curves: list
    :return:
    """
    # Output folder to save simulation results.
    output_plot_dir = osp.dirname(__file__) + '/' + 'output'
    if not osp.exists(output_plot_dir):
        mkdir(output_plot_dir)

    # save velocity versus time plot
    fig1 = plt.figure('Velocity-Time')
    plt.plot(times, velocities[1:])
    plt.grid()
    fig1.savefig(output_plot_dir + '/' +
                 f"veh_{starting_velocity}_{vehicle_id}_{gear_shifting_style}_{driver_style}_time_velocity_plot.png", dpi=150)

    # save acceleration versus velocity plot
    fig2 = plt.figure('Acceleration-Velocity')
    plt.plot(velocities[1:], accelerations[1:])
    plt.grid()
    fig2.savefig(output_plot_dir + '/' +
                 f"veh_{starting_velocity}_{vehicle_id}_{gear_shifting_style}_{driver_style}_velocity_acceleration_plot.png", dpi=150)

    # save position versus time plot
    fig3 = plt.figure('Position-Time')
    plt.plot(times, positions[1:])
    plt.grid()
    fig3.savefig(output_plot_dir + '/' +
                 f"veh_{starting_velocity}_{vehicle_id}_{gear_shifting_style}_{driver_style}_acceleration_time_plot.png", dpi=150)

    # save acceleration versus time plot
    fig4 = plt.figure('Acceleration-Time')
    plt.plot(times, accelerations[1:])
    plt.grid()
    fig4.savefig(output_plot_dir + '/' +
                 f"veh_{starting_velocity}_{vehicle_id}_{gear_shifting_style}_{driver_style}_acceleration_time_plot.png", dpi=150)

    plt.figure('Acceleration-Velocity')
    for i, gear_curve in enumerate(curves):
        sp_bins = np.arange(start[i], stop[i] + 0.1, 0.1)
        accelerations = gear_curve(sp_bins)
        plt.plot(sp_bins, accelerations, 'k')
    plt.grid()
    plt.show()

    return 0


if __name__ == '__main__':
    plot_and_save_simulation_result()