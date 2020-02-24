from os import path as osp, chdir
from co2mpas_driver import dsp as driver
import numpy as np
import matplotlib.pyplot as plt

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    Vehicle simulation.

    :return:
    """

    # A sample car id from the database
    car_id = 8188
    # The gear shifting style as described in the TRR paper.
    gs_style = 0.99

    # The desired speed
    vdes = 124/3.6

    # Current speed
    v_start = 0

    # The simulation step in seconds
    sim_step = 0.1

    # The driving style as described in the TRR paper.
    driver_style = 0.2

    # Duration of the simulation in seconds.
    duration = 100

    # sample time series
    times = np.arange(0, duration + sim_step, sim_step)

    sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
        gear_shifting_style=gs_style, desired_velocity=vdes,
        starting_velocity=v_start, degree=4, driver_style=driver_style,
        sim_start=0, sim_step=sim_step, duration=duration, use_linear_gs=True,
        use_cubic=False))))[
        'outputs']
    velocities = sol['velocities']
    accelerations = sol['accelerations']
    # driver.plot(1)
    plt.figure('Speed-Acceleration')
    plt.plot(velocities[1:], accelerations[1:])
    plt.grid()

    plt.figure('Time-Speed')
    plt.plot(times, velocities[1:])
    plt.grid()
    plt.figure('Acceleration-Time')
    plt.plot(times, accelerations[1:])
    plt.grid()

    plt.show()
    return 0


if __name__ == '__main__':
    simple_run()
