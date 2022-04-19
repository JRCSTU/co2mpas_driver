from os import path as osp, chdir
from co2mpas_driver.common import simulation_part as sp
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno
from co2mpas_driver.common import gear_functions as fg

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    This example performs a simulation with MFC model from a starting speed to a desired speed.

    :return:
    """
    # Vehicle databased based on the Euro Car Segment classification
    db_path = osp.abspath(
        osp.join(osp.dirname(my_dir + "/../"), "co2mpas_driver", "db", "EuroSegmentCar")
    )

    # A sample car id from the database
    car_id = 47844

    # The gear shifting style as described in the TRR paper.
    gs_style = 0.9

    # The desired speed
    vdes = 124 / 3.6

    # Current speed
    v_start = 0

    # The simulation step in seconds
    sim_step = 0.1

    # The driving style as described in the TRR paper.
    driver_style = 0.2

    # Duration of the simulation in seconds.
    duration = 100

    # sample time series
    times = np.arange(10, duration + sim_step, sim_step)

    # load vehicle database
    db = rno.load_db_to_dictionary(db_path)

    # The vehicle specs as returned from the database
    selected_car = rno.get_vehicle_from_db(db, car_id, electric=True)

    """
    The final acceleration curvers (Curves), the engine acceleration potential
    curves (cs_acc_per_gear), before the calculation of the resistances and the
    limitation due to max possible acceleration (friction) .
    """
    Curves, Curves_dec, StartStop, gs = mf.get_ev_curve_main(selected_car)

    # Curves, cs_acc_per_gear, StartStop, gs = mf.gear_curves_n_gs_from_
    # poly(selected_car, gs_style,4)

    """Lists to gather simulation data"""
    Speeds = [v_start]
    Acceleration = [0]

    """Initialize speed and gear"""
    speed = v_start
    """
    Returns the gear that must be used and the clutch condition
    """
    gear, gear_count = fg.gear_for_speed_profiles(gs, speed, 0, 0)
    gear_count = 0

    """Core loop"""
    for t in times:
        speed, acceleration, gear, gear_count = sp.simulation_step_function(
            selected_car,
            speed,
            gear,
            gear_count,
            gs,
            Curves,
            vdes,
            driver_style,
            sim_step,
        )

        """Gather data"""
        Speeds.append(speed)
        Acceleration.append(acceleration)

    """Plot"""
    fig2 = plt.figure()
    plt.plot(times, Speeds[1:])
    fig2.suptitle("Speed over time", x=0.54, y=1, fontsize=12)
    plt.xlabel("Time (sec)", fontsize=12)
    plt.ylabel("Speed (m/s)", fontsize=12)
    plt.grid()

    fig3 = plt.figure("Acceleration over Speed")
    plt.plot(Speeds[1:], Acceleration[1:])
    fig3.suptitle("Acceleration over Speed", x=0.54, y=1, fontsize=12)
    plt.xlabel("Speed (m/s)", fontsize=12)
    plt.ylabel("Acceleration (m/s2)", fontsize=12)
    plt.grid()

    fig4 = plt.figure("Acceleration-Time")
    plt.plot(times, Acceleration[1:])
    fig4.suptitle("Acceleration over Time", x=0.54, y=1, fontsize=12)
    plt.xlabel("Time (sec)", fontsize=12)
    plt.ylabel("Acceleration (m/s2)", fontsize=12)
    plt.grid()

    fig5 = plt.figure("Acceleration over Speed")
    for i, gear_curve in enumerate(Curves):
        sp_bins = np.arange(StartStop[0][i], StartStop[1][i] + 0.1, 0.1)
        accelerations = gear_curve(sp_bins)
        plt.plot(sp_bins, accelerations, "k")
    fig5.suptitle("Acceleration over Speed", x=0.54, y=1, fontsize=12)
    plt.xlabel("Speed (m/s)", fontsize=12)
    plt.ylabel("Acceleration (m/s2)", fontsize=12)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simple_run()
