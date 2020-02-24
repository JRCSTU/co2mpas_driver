from os import path as osp, chdir
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    car_id = 40516
    gs_style = 0.8
    degree = 4

    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
        gear_shifting_style=gs_style, degree=degree, use_linear_gs=True,
        use_cubic=False))))[
        'outputs']

    # start = sol['start']  # Start/stop speed for each gear
    # stop = sol['stop']
    # sp_bins = sol['sp_bins']
    # full_load_speeds = sol['full_load_speeds']  # Full load curves of speed
    # and torque
    # full_load_torque = sol['full_load_torque']
    # speed_per_gear = sol['speed_per_gear']  # speed and acceleration ranges
    # and poitns for each gear
    # acc_per_gear = sol['acc_per_gear']
    # car_res_curve = sol['car_res_curve']  # get resistances
    # car_res_curve_force = sol['car_res_curve_force']
    # curves = sol['curves']
    # gs = sol['gs']
    # poly_spline = sol['poly_spline']  # extract speed acceleration Splines
    discrete_acceleration_curves = sol['discrete_acceleration_curves']
    # driver.plot(1)
    for curve in discrete_acceleration_curves:
        sp_bins = list(curve['x'])
        acceleration = list(curve['y'])
        plt.plot(sp_bins, acceleration)
    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
