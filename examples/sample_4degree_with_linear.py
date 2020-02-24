from os import path as osp, chdir
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    Example of how to extract the acceleration curves of a vehicle and the
    corresponding plot. The final acceleration curvers (Curves), the engine
    acceleration potential curves (poly_spline), before the calculation of the
    resistances and the limitation due to max possible acceleration
    (friction).

    """
    # A sample car id from the database
    car_id = 35135
    # The gear shifting style as described in the TRR paper.
    gs_style = 0.8

    # How to use co2mpas_driver library
    # You can also pass vehicle database path db_path='path to vehicle db'
    sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
        gear_shifting_style=gs_style, degree=4, use_linear_gs=True,
        use_cubic=False))))[
        'outputs']
    # driver.plot(1)
    # poly_spline = sol['poly_spline']
    for curve in sol['discrete_acceleration_curves']:
        sp_bins = list(curve['x'])
        acceleration = list(curve['y'])
        plt.plot(sp_bins, acceleration, 'o')
    '''import vehicle object, curves and gear shifting strategy'''

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
