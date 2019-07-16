from new_MFC import vehicle_functions as vf
import numpy as np
from new_MFC.common import gear_functions as fg
import matplotlib.pyplot as plt


def gear_curves(my_car):
    """Full load curves of speed and torque"""
    full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(my_car)

    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torque)

    """Extract speed acceleration Cubic Splines"""
    poly_spline = vf.get_cubic_splines_of_speed_acceleration_relationship(
        my_car, speed_per_gear, acc_per_gear)

    """Start/stop speed for each gear"""
    Start, Stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.1)

    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """Calculate Curves"""
    Curves = vf.calculate_curves_to_use(poly_spline, Start, Stop, Alimit,
                                        car_res_curve, sp_bins)

    return Curves, poly_spline, (Start, Stop)


def gear_curves_n_gs(my_car, gs_style, degree):
    """
    Not used in the last version of MFC (for reference).
    Full load curves of speed and torque
    """
    full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(my_car)

    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torque)

    """Extract speed acceleration Cubic Splines"""
    cs_acc_per_gear = vf.get_cubic_splines_of_speed_acceleration_relationship(
        my_car, speed_per_gear, acc_per_gear)

    """Start/stop speed for each gear"""
    Start, Stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    cs_acc_per_gear)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.1)
    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """Calculate Curves"""
    Curves = vf.calculate_curves_to_use(cs_acc_per_gear, Start, Stop, Alimit,
                                        car_res_curve, sp_bins)

    """Get gs"""
    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, degree)
    Tans = fg.find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop)
    gs = fg.gear_points_from_tan(Tans, gs_style, Start, Stop)

    return Curves, cs_acc_per_gear, (Start, Stop), gs


def gear_curves_n_gs_from_poly(my_car, gs_style, degree):
    """Full load curves of speed and torque"""
    full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(my_car)

    """Speed and acceleration ranges and points for each gear"""
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        my_car, full_load_speeds, full_load_torque)

    """Extract speed acceleration Splines"""
    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, degree)
    starting_speed = vf.get_starting_speed(coefs_per_gear)
    poly_spline = vf.get_spline_out_of_coefs(coefs_per_gear, starting_speed)

    """Start/stop speed for each gear"""
    Start, Stop = vf.get_start_stop(my_car, speed_per_gear, acc_per_gear,
                                    poly_spline)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.1)
    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """Calculate Curves"""
    Curves = vf.calculate_curves_to_use(poly_spline, Start, Stop, Alimit,
                                        car_res_curve, sp_bins)

    Tans = fg.find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop)
    gs = fg.gear_points_from_tan(Tans, gs_style, Start, Stop)

    return Curves, poly_spline, (Start, Stop), gs


def gear_4degree_curves_with_linear_gs(my_car, gs_style):
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

    return Curves, poly_spline, (Start, Stop), gs


def get_ev_curve_main(my_car):
    """Full load curves of speed and torque"""
    cs_acc_per_gear, Start, Stop = vf.ev_curve(my_car)

    sp_bins = np.arange(0, Stop[-1] + 1, 0.1)
    """Get resistances"""
    car_res_curve, car_res_curve_force, Alimit = vf.get_resistances(my_car,
                                                                    sp_bins)

    """Calculate Curves"""
    Curves = vf.calculate_curves_to_use(cs_acc_per_gear, Start, Stop, Alimit,
                                        car_res_curve, sp_bins)

    return Curves, (Start, Stop)
