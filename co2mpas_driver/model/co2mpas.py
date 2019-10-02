# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to process a CO2MPAS input file.
"""

# Computation of the MFC vehicle acceleration - speed curve.
# coding=utf-8
import functools
import numpy as np
from scipy.interpolate import CubicSpline
from co2mpas_driver.common import defaults as defaults


def get_full_load(ignition_type):
    """
    Get vehicle full load curve.

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
    Calculate Full load curves of speed and torque.

    :param full_load_speeds:
        Full load speeds.
    :type full_load_powers: str

    :param full_load_powers:
        Engine nominal power [kW].
    :type full_load_speeds: float

    :return: full_load_torques
        Full load torques.
    :rtype:
    """
    full_load_torques = full_load_powers * 1000 * (
            full_load_speeds / 60 * 2 * np.pi) ** -1

    return full_load_torques


# The maximum acceleration that the vehicle can have on the road
def Armax(car_type, veh_mass, engine_max_power, road_type=1):
    """
    Calculating the maximum acceleration possible for the vehicle object my_car,
    under road_type conditions.

    :param car_type:
        Car type.
    :type car_type: int

    :param veh_mass:
        Vehicle mass.
    :type veh_mass: float

    :param engine_max_power:
        Maximum engine power.
    :type engine_max_power:

    :param road_type:
        Road type(1: normal, 2: wet, 3: icy)
    :type road_type: int

    :return:
        Maximum possible acceleration of the vehicle.
    :rtype:
    """
    if car_type == 2:  # forward-wheel drive vehicles
        f_mass = 0.6 * veh_mass
    elif car_type == 4:  # rear-wheel drive vehicles
        f_mass = 0.45 * veh_mass
    else:  # all-wheel drive vehicles, 4x4
        f_mass = 1 * veh_mass

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
    # maximum achieved force by the vehicles in certain conditions.
    f_max = f_mass * 9.8066 * mh

    return f_max / veh_mass


# Calculates a spline with the resistances
def veh_resistances(f0, f1, f2, sp_bins, total_mass):
    """
    Calculate the resistances that a vehicle faces, per speed.

    :param f0:
        Tire rolling resistance.
    :type f0: float

    :param f1:
        Partly tire rolling resistance & partly drivetrain losses.
    :type f1: float

    :param f2:
        Aerodynamic component (proportional to the square of the vehicles
        velocity)
    :type f2: float

    :param sp_bins:
        Speed bins.
    :type sp_bins: list[float]

    :param total_mass:
        Total mass.
    :type total_mass: float

    :return:
        Resistance forces being applied per speed.
    :rtype:
    """
    sp_bins = list(sp_bins)
    resistance_force = []
    for i in range(len(sp_bins)):
        resistance_force.append(f0 + f1 * sp_bins[i] * 3.6 + f2 * pow(sp_bins[i] * 3.6, 2))
        # Facc = Fmax @ wheel - f0 * cos(a) - f1 * v - f2 * v2 - m * g * sin(a)

    approximate_mass = int(total_mass)
    acc_resistance = [x / approximate_mass for x in resistance_force]
    a = int(np.floor(sp_bins[0]))
    b = int(np.floor(sp_bins[-1]))
    resistance_spline_curve = CubicSpline(
        [k for k in range(a - 10, a)] + sp_bins + [k for k in range(b + 1, b + 11)], \
        [acc_resistance[0]] * 10 + acc_resistance + [acc_resistance[-1]] * 10)
    resistance_spline_curve_f = CubicSpline(
        [k for k in range(a - 10, a)] + sp_bins + [k for k in range(b + 1, b + 11)],
        [resistance_force[0]] * 10 + resistance_force + [resistance_force[-1]] * 10)

    return resistance_spline_curve, resistance_spline_curve_f


def estimate_f_coefficients(veh_mass, type_of_car, car_width, car_height,
                            passengers=0):
    """
    Estimate f0, f1, f2 coefficients of resistance.

    :param veh_mass:
        Vehicle mass.
    :type veh_mass: float

    :param type_of_car:
        Type of car.
    :type type_of_car: str

    :param car_width:
        Car width.
    :type car_width: float

    :param car_height:
        Car height.
    :type car_height: float

    :param passengers:
        number of passengers.
    :type passengers: int

    :return:
        Coefficients of resistance force.
    :rtype: float, float, float
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
