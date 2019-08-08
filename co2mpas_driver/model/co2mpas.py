# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to processes a CO2MPAS input file.
"""

# Computation of the MFC vehicle acceleration - speed curve.
# coding=utf-8
import functools as functools
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from co2mpas_driver.common import defaults as defaults


def get_full_load(ignition_type):
    """
    Returns vehicle full load curve.

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

    :return:
        Vehicle normalized full load curve.
    :rtype: scipy.interpolate.InterpolatedUnivariateSpline
    """

    xp, fp = defaults.dfl.functions.get_full_load.FULL_LOAD[ignition_type]
    func = functools.partial(
        np.interp, xp=xp, fp=fp, left=fp[0], right=fp[-1]
    )

    return func


def calculate_full_load_speeds_and_powers(
        full_load_curve, engine_max_power, engine_max_speed_at_max_power,
        idle_engine_speed):
    """
    Calculates the full load speeds and powers [RPM, kW].

    :param full_load_curve:
        Vehicle normalized full load curve.
    :type full_load_curve: scipy.interpolate.InterpolatedUnivariateSpline

    :param engine_max_power:
        Engine nominal power [kW].
    :type engine_max_power: float

    :param engine_max_speed_at_max_power:
        Engine nominal speed at engine nominal power [RPM].
    :type engine_max_speed_at_max_power: float

    :param idle_engine_speed:
        Engine speed idle median and std [RPM].
    :type idle_engine_speed: (float, float)

    :return:
         T1 map speed [RPM] and power [kW] vectors.
    :rtype: (numpy.array, numpy.array)
    """

    n_norm = np.arange(0.0, 1.21, 0.1)
    full_load_powers = full_load_curve(n_norm) * engine_max_power
    idle = idle_engine_speed[0]
    full_load_speeds = n_norm * (engine_max_speed_at_max_power - idle) + idle

    return full_load_speeds, full_load_powers


def calculate_full_load_torques(full_load_speeds, full_load_powers):
    """
    Full load curves of speed and torque.

    :param full_load_powers:
        Engine ignition type (positive or compression).
    :type full_load_powers: str

    :param full_load_speeds:
        Engine nominal power [kW].
    :type full_load_speeds: float
    :return: full_load_torques
    """
    full_load_torques = full_load_powers * 1000 * (
            full_load_speeds / 60 * 2 * np.pi) ** -1

    return full_load_torques


# The maximum force that the vehicle can have on the road
def Armax(car_type, veh_mass, engine_max_power, road_type=1):
    """

    Calculating the maximum acceleration possible for the vehicle object my_car, under road_type conditions

    :param car_type:
    :param veh_mass:
    :param engine_max_power:
    :param road_type: road condition (1: normal, 2: wet, 3: icy)
    :return:
    """
    if car_type == 2:  # forward-wheel drive vehicles
        fmass = 0.6 * veh_mass
    elif car_type == 4:  # rear-wheel drive vehicles
        fmass = 0.45 * veh_mass
    else:  # all-wheel drive vehicles, 4x4
        fmass = 1 * veh_mass

    if road_type == 1:
        mh_base = 0.75  # for normal road
    elif road_type == 2:
        mh_base = 0.25  # for wet road
    else:
        mh_base = 0.1  # for icy road
    # Optimal values:
    # 0.8 dry, 0.6 light rain, 0.4 heavy rain, 0.1 icy
    alpha = 43.398
    beta = 5.1549
    mh = mh_base * (alpha * np.log(engine_max_power) + beta) / 190

    # * cos(f) for the gradient of the road. Here we consider as 0
    Frmax = fmass * 9.8066 * mh

    return Frmax / veh_mass


# Calculates a spline with the resistances
def veh_resistances(f0, f1, f2, sp, total_mass):
    """
    Return the resistances that a vehicle faces, per speed

    :param f0:
    :param f1:
    :param f2:
    :param sp:
    :param total_mass:
    :return:
    """
    sp = list(sp)
    Fresistance = []
    for i in range(len(sp)):
        Fresistance.append(f0 + f1 * sp[i] * 3.6 + f2 * pow(sp[i] * 3.6, 2))
        # Facc = Fmax @ wheel - f0 * cos(a) - f1 * v - f2 * v2 - m * g * sin(a)

    aprx_mass = int(total_mass)
    Aresistance = [x / aprx_mass for x in Fresistance]
    a = int(np.floor(sp[0]))
    b = int(np.floor(sp[-1]))
    resistance_spline_curve = CubicSpline(
        [k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)], \
        [Aresistance[0]] * 10 + Aresistance + [Aresistance[-1]] * 10)
    resistance_spline_curve_f = CubicSpline(
        [k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)],
        [Fresistance[0]] * 10 + Fresistance + [Fresistance[-1]] * 10)

    return resistance_spline_curve, resistance_spline_curve_f


def estimate_f_coefficients(veh_mass, type_of_car, car_width, car_height,
                            passengers=0):
    """
    f0, f1, f2 coefficients of resistance are estimated

    :param veh_mass:
    :param type_of_car:
    :param car_width:
    :param car_height:
    :param passengers:
    :return:
    """

    d = {}
    # Fill in the entries one by one
    d["cabriolet"] = 0.28
    d["sedan"] = 0.27
    d["hatchback"] = 0.3
    d["stationwagon"] = 0.28
    d["suv/crossover"] = 0.35
    d["mpv"] = 0.3
    d["coupé"] = 0.27
    d["pick-up"] = 0.4

    rolling_res_coef = 0.009  # Constant for the moment
    theor_aero_coeff = d[type_of_car]

    operating_mass = veh_mass + 100 + 75 * passengers
    f0 = (operating_mass + 100) * rolling_res_coef * 9.81
    f2 = 0.5 * 1.2 * (
            0.84 * car_width * car_height * theor_aero_coeff) / pow(
        3.6, 2)
    f1 = -71.735 * f2 + 2.7609

    return f0, f1, f2
