import matplotlib.pyplot as plt
import numpy as np


def plot_speed_acceleration_from_coefs(
    poly_coefs_per_gear, speeds_per_gear, accelerations_per_gear
):
    """
    Plot the speed acceleration diagram created.

    :param poly_coefs_per_gear: Polynomial coefficients for the acceleration over speed curves
    :param speeds_per_gear: Speed points
    :param accelerations_per_gear: Acceleration points
    :return:
    """

    degree = len(poly_coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    plt.figure("speed acceleration regression results of degree = " + str(degree))

    for speeds, acceleration, fit_coef in zip(
        speeds_per_gear, accelerations_per_gear, poly_coefs_per_gear
    ):
        plt.plot(speeds, acceleration, "kx")

        # x_new = np.linspace(speeds[0], speeds[-1], 100)
        x_new = np.arange(speeds[0], speeds[-1], 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars)) for i in x_new])

        plt.plot(x_new, a_new, "rx")
