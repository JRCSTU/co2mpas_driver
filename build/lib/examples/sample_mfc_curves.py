from os import path as osp, chdir
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.model import define_discrete_poly as ddp
from co2mpas_driver.model import define_discrete_car_res_curve as ddc
from co2mpas_driver.model import define_discrete_car_res_curve_force as ddcf
from co2mpas_driver.common import reading_n_organizing as rno
from co2mpas_driver.common import vehicle_functions as vf
from co2mpas_driver.common import gear_functions as fg

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    # file path without extension of the file
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar'))
    car_id = 40516
    gs_style = 0.8
    # degree = 4

    db = rno.load_db_to_dictionary(db_path)
    my_car = rno.get_vehicle_from_db(db, car_id)

    """Full load curves of speed and torque"""
    full_load_speeds, full_load_torques = vf.get_load_speed_n_torque(my_car)

    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torques)

    """Extract speed acceleration Splines"""
    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, 4)
    poly_spline = vf.get_spline_out_of_coefs(coefs_per_gear,
                                             speed_per_gear[0][0])

    """Start/stop speed for each gear"""
    start, stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, stop[-1] + 1, 0.01)

    """define discrete poly spline"""
    discrete_poly_spline = ddp(poly_spline, sp_bins)

    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """define discrete_car_res_curve"""
    discrete_car_res_curve = ddc(car_res_curve, sp_bins)

    """define discrete_car_res_curve_force"""
    discrete_car_res_curve_force = ddcf(car_res_curve_force, sp_bins)

    """Calculate Curves"""
    curves = vf.calculate_curves_to_use(poly_spline, start, stop, Alimit,
                                        car_res_curve, sp_bins)

    """Get gs"""
    gs = fg.gear_linear(speed_per_gear, gs_style)

    from co2mpas_driver.model import define_discrete_acceleration_curves as func
    discrete_acceleration_curves = func(curves, start, stop)
    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
