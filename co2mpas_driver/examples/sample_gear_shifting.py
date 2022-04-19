from os import path as osp, chdir
import matplotlib.pyplot as plt
import numpy as np
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    This sample file plots the speed/acceleration curves of the vehicle and
    regression results of 2nd degree polynomial based on gear shifting style.

    :return:
    """
    # A sample car id from the database
    car_id = 27748

    # The gear shifting style as described in the TRR paper.
    gs_style = 1

    # 2nd degree polynomial
    degree = 2

    # Core model, this will select and execute the proper functions for the given inputs and returns the output
    # You can also pass your vehicle's database path db_path='path to vehicle db'
    sol = driver(
        dict(
            vehicle_id=car_id,
            inputs=dict(
                inputs=dict(
                    gear_shifting_style=gs_style,
                    gedree=degree,
                    use_linear_gs=False,
                    use_cubic=True,
                )
            ),
        )
    )["outputs"]
    # Plots workflow of the core model, this will automatically open an internet browser and shows the work flow
    # of the core model. you can click all the rectangular boxes to see in detail sub models like load, model,
    # write and plot.
    # you can uncomment to plot the work flow of the model
    # driver.plot(1)  # Note: run your IDE as Administrator if file permission error.

    # gear cuts based on linear gear shifting style
    gs = sol["gs"]

    # coefficients of speed acceleration relation for each gear
    # calculated using 2nd degree polynomial
    coefs_per_gear = sol["coefs_per_gear"]
    speed_per_gear = sol["speed_per_gear"]
    acc_per_gear = sol["acc_per_gear"]

    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    fig = plt.figure("speed acceleration regression results of degree = " + str(degree))
    for gear in gs:
        plt.plot([gear, gear], [0, 5])

    for gear, (speeds, acceleration, fit_coef) in enumerate(
        zip(speed_per_gear, acc_per_gear, coefs_per_gear)
    ):
        plt.plot(speeds, acceleration, label=f"gear {gear}")
        x_new = np.arange(speeds[0], speeds[-1], 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars)) for i in x_new])
        plt.plot(x_new, a_new, label=f"{degree} degree reg. gear {gear}")
    plt.legend()
    plt.text(46, 1.8, f"Simulation car:{car_id}, GS:{gs_style}")
    fig.suptitle(
        "Speed acceleration regression results of degree = " + str(degree),
        x=0.54,
        y=1,
        fontsize=12,
    )
    plt.xlabel("Speed (m/s)", fontsize=12)
    plt.ylabel("Acceleration (m/s2)", fontsize=12)
    plt.show()


if __name__ == "__main__":
    simple_run()
