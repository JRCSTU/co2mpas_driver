import os
import numpy as np
import matplotlib.pyplot as plt
import curve_functions as mf
import reading_n_organizing as rno
import vehicle_functions as vf
import gear_functions as fg


def simple_run():
    # db_name = '../db/car_db_sample'
    db_name = '../db/EuroSegmentCar_cleaned'
    car_id = 40516
    gs_style = 0.8
    # degree = 4

    # file path without extension of the file
    db_name = os.path.dirname(db_name) + '/' + \
              os.path.splitext(os.path.basename(db_name))[0]

    db = rno.load_db_to_dictionary(db_name)
    my_car = rno.get_vehicle_from_db(db, car_id)

    """Full load curves of speed and torque"""
    full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(my_car)

    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torque)

    """Extract speed acceleration Splines"""
    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, 4)
    poly_spline = vf.get_spline_out_of_coefs(coefs_per_gear,
                                             speed_per_gear[0][0])

    """Start/stop speed for each gear"""
    Start, Stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.01)
    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """Calculate Curves"""
    Curves = vf.calculate_curves_to_use(poly_spline, Start, Stop, Alimit,
                                        car_res_curve, sp_bins)

    """Get gs"""
    gs = fg.gear_linear(speed_per_gear, gs_style)

    return 0


if __name__ == '__main__':
    simple_run()
