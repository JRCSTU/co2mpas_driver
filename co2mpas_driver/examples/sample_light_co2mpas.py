from os import path as osp, chdir
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    This example computes and plots the CO2 emissions in grams for a
    simulated trajectory.

    """
    car_id = 35135

    # The gear shifting style as described in the TRR paper.
    gs_style = 0.8

    # The desired speed
    vdes = 124 / 3.6

    # Current speed
    v_start = 0

    # The simulation step in seconds
    sim_step = 0.1

    # The driving style as described in the TRR paper.
    driver_style = 0.6

    # Duration of the simulation in seconds.
    duration = 100

    # sample time series
    times = np.arange(0, duration + sim_step, sim_step)

    # core model, this will select and execute the proper functions for the given inputs and returns the output
    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(
        dict(
            vehicle_id=car_id,
            inputs=dict(
                inputs=dict(
                    gear_shifting_style=gs_style,
                    desired_velocity=vdes,
                    starting_velocity=v_start,
                    driver_style=driver_style,
                    sim_start=0,
                    sim_step=sim_step,
                    duration=duration,
                    degree=4,
                    use_linear_gs=True,
                    use_cubic=False,
                )
            ),
        )
    )["outputs"]
    # Plots workflow of the core model, this will automatically open an internet browser and show the work flow
    # of the core model. you can click all the rectangular boxes to see in detail sub models like load, model,
    # write and plot.
    # you can uncomment to plot the work flow of the model
    # driver.plot(1)  # Note: run your IDE as Administrator if file permission error.

    driver_simulation_model = sol["driver_simulation_model"]

    res = {}
    for myt in times:
        if myt == times[0]:
            driver_simulation_model.reset(v_start)
            res = {"accel": [0], "speed": [v_start], "position": [0], "gear": [0]}
            continue
        gear, next_velocity, acc, position = driver_simulation_model(sim_step, vdes)
        res["accel"].append(acc)
        res["speed"].append(next_velocity)
        res["gear"].append(gear)
        res["position"].append(position)

    fig = plt.figure("Speed-Acceleration")
    plt.plot(
        res["speed"],
        res["accel"],
        label=f"Simulation car:{car_id}, DS: {driver_style}, GS:{gs_style}",
    )
    plt.legend()
    fig.suptitle("Acceleration over Speed", x=0.54, y=1, fontsize=12)
    plt.xlabel("Speed (m/s)", fontsize=14)
    plt.ylabel("Acceleration (m/s2)", fontsize=12)
    plt.grid()
    plt.show()

    fp = sol["fp"]

    fig = plt.figure("Speed-co2")
    plt.plot(
        res["speed"],
        fp,
        label=f"Simulation car:{car_id}, DS: {driver_style}, GS:{gs_style}",
    )
    plt.legend()
    fig.suptitle("Co2 emission for a simulated trajectory", x=0.54, y=1, fontsize=12)
    plt.xlabel("Speed (m/s)", fontsize=12)
    plt.ylabel("Co2 (gms)", fontsize=12)
    plt.show()


if __name__ == "__main__":
    simple_run()
