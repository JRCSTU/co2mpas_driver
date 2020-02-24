from os import path as osp, chdir
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    car_id = 39393
    gs_style = 0.8  # gear shifting can take value from 0(timid driver)
    degree = 2

    # How to use co2mpas_driver library
    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
        gear_shifting_style=gs_style, degree=degree, use_linear_gs=True,
        use_cubic=False))))[
        'outputs']
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
