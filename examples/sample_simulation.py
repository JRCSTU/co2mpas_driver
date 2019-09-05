from os import chdir, path as osp
from co2mpas_driver.common import simulation_part as sp
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno
from co2mpas_driver.common import gear_functions as fg

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """:parameters of the simulation"""
    # Vehicle databased based on the Euro Car Segment classification
    # file path without extension of the file
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar'))
    # A sample car id from the database
    car_id = 35135
    # The gear shifting style as described in the TRR paper.
    gs_style = 0.7

    # The desired speed
    v_des = 40

    # Current speed
    v_start = 0

    # The simulation step in seconds
    sim_step = 0.1

    # The driving style as described in the TRR paper.
    driver_style = 1

    # Duration of the simulation in seconds.
    duration = 100

    # sample time series
    times = np.arange(0, duration + sim_step, sim_step)

    '''import vehicle object, curves and gear shifting strategy'''
    db = rno.load_db_to_dictionary(db_path)

    # The vehicle specs as returned from the database
    selected_car = rno.get_vehicle_from_db(db, car_id)

    # ***********************************************************************
    """
    The final acceleration curvers (Curves), the engine acceleration potential 
    curves (cs_acc_per_gear), before the calculation of the resistances and the
    limitation due to max possible acceleration (friction) .
    """
    curves, cs_acc_per_gear, start_stop, gs = \
        mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)

    """
    The difference betweeen "gear_4degree_curves_with_linear_gs" and 
    "gear_curves_n_gs_from_poly" is the computation of the engine acceleration 
    potential curves
    """
    # Curves, cs_acc_per_gear, StartStop, gs = mf.gear_curves_n_gs_from_poly(
    # selected_car, gs_style, 4)

    # ********* define function for gathering simulation data ****************
    """Lists to gather simulation data"""
    speeds = [v_start]
    acceleration = [0]

    """Initialize speed and gear"""
    speed = v_start
    """
    Returns the gear that must be used and the clutch condition
    """
    gear, gear_count = fg.gear_for_speed_profiles(gs, speed, 0, 0)
    gear_count = 0
    gears = [gear]
    """Core loop"""
    for t in np.diff(times):
        speed, gear, gear_count = sp.simulation_step_function(
            selected_car.transmission, speed, gear, gear_count, gs, curves,
            v_des, driver_style, sim_step)

        """Gather data"""
        gears.append(gear)
        speeds.append(speed)
        acceleration.append((speeds[-1] - speeds[-2]) / sim_step)
    # ******************* Plot*************************
    """Plot"""
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
    for i, gear_curve in enumerate(curves):
        sp_bins = np.arange(start_stop[0][i], start_stop[1][i] + 0.1, 0.1)
        accelerations = gear_curve(sp_bins)
        plt.plot(sp_bins, accelerations, 'k')
    plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    simple_run()
