from os import path as osp, chdir
import matplotlib.pyplot as plt
import numpy as np
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    car_id = 27748
    gs_style = 1
    degree = 2

    # How to use co2mpas_driver library
    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
        gear_shifting_style=gs_style, gedree=degree, use_linear_gs=False,
        use_cubic=True))))[
        'outputs']
    # driver.plot(1)
    # gs = sol['gs']

    coefs_per_gear = sol['coefs_per_gear']
    speed_per_gear = sol['speed_per_gear']
    acc_per_gear = sol['acc_per_gear']

    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    plt.figure('speed acceleration regression results of degree = ' + str(degree))

    for speeds, acceleration, fit_coef in zip(speed_per_gear, acc_per_gear,
                                              coefs_per_gear):
        plt.plot(speeds, acceleration)

        # x_new = np.linspace(speeds[0], speeds[-1], 100)
        x_new = np.arange(speeds[0], speeds[-1], 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars)) for i in x_new])

        plt.plot(x_new, a_new)

    plt.show()


if __name__ == '__main__':
    simple_run()