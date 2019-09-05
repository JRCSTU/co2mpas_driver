from co2mpas_driver.common import vehicle_functions as vf
from co2mpas_driver.common import gear_functions as fg
from co2mpas_driver.common import plot_templates as pt
from co2mpas_driver.model import define_discrete_poly as ddp

from os import path as osp, chdir
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.common import reading_n_organizing as rno

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar'))
    car_id = 27748
    gs_style = 1

    db = rno.load_db_to_dictionary(db_path)

    selected_car = rno.get_vehicle_from_db(db, car_id)

    full_load_speeds, full_load_torques = vf.get_load_speed_n_torque(selected_car)
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        selected_car, full_load_speeds, full_load_torques)

    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, 2)
    pt.plot_speed_acceleration_from_coefs(coefs_per_gear, speed_per_gear,
                                          acc_per_gear)

    poly_spline = vf.get_cubic_splines_of_speed_acceleration_relationship(
        selected_car, speed_per_gear, acc_per_gear)

    Start, Stop = vf.get_start_stop(selected_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.01)
    """define discrete poly spline"""
    discrete_poly_spline = ddp(poly_spline, sp_bins)

    tans = fg.find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop)

    gs = fg.gear_points_from_tan(tans, gs_style, Start, Stop)

    for gear in gs:
        plt.plot([gear, gear], [0, 5], 'k')

    plt.show()


if __name__ == '__main__':
    simple_run()