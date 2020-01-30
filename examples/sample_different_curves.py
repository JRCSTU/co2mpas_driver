from os import path as osp, chdir
import matplotlib.pyplot as plt
import numpy as np
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    # Vehicles database path
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar'))
    car_id = 39393
    gs_style = 0.8  # gear shifting can take value from 0(timid driver)
    degree = 2

    db = rno.load_db_to_dictionary(db_path)
    selected_car = rno.get_vehicle_from_db(db, car_id)

    curves, cs_acc_per_gear, start_stop, gs = mf.gear_curves_n_gs_from_poly(
        selected_car, gs_style, degree)

    from co2mpas_driver.model import define_discrete_acceleration_curves as func, \
        define_discrete_poly as func_, define_discrete_car_res_curve as rc, get_resistances as gr, \
        define_discrete_car_res_curve_force as cf, define_discrete_acceleration_curves as ac
    sp_bins = np.arange(0, start_stop[1][-1] + 1, 0.1)
    discrete_acceleration_curves = func(curves, *start_stop)
    discrete_poly_spline = func_(cs_acc_per_gear, sp_bins)
    car_res_curve, car_res_curve_force = gr(selected_car.type_of_car,
                                            selected_car.veh_mass,
                                            selected_car.car_width,
                                            selected_car.car_height, sp_bins)
    discrete_car_res_curve = rc(car_res_curve, sp_bins)
    discrete_car_res_curve_force = cf(car_res_curve_force, sp_bins)
    discrete_acceleration_curves = ac(curves, *start_stop)
    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
