# -*- coding: utf-8 -*-
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to process a CO2MPAS input file.

Sub-Modules:
.. currentmodule:: co2mpas_driver.model
.. autosummary::
:nosignatures:
:toctree: model/

    co2mpas
    simulation
"""
import numpy as np
import schedula as sh
from .driver import Driver as dr
from co2mpas_driver.model.co2mpas import (
    get_full_load,
    calculate_full_load_speeds_and_powers,
    calculate_full_load_torques,
)

dsp = sh.Dispatcher(name="model")
dsp.add_func(get_full_load, outputs=["full_load_curve"])
dsp.add_func(
    calculate_full_load_speeds_and_powers,
    outputs=["full_load_speeds", "full_load_powers"],
)
dsp.add_func(calculate_full_load_torques, outputs=["full_load_torques"])


# Speed and acceleration ranges and points for each gear
@sh.add_function(dsp, outputs=["speed_per_gear", "acc_per_gear"])
def get_speeds_n_accelerations_per_gear(
    gear_box_ratios,
    idle_engine_speed,
    tyre_radius,
    driveline_slippage,
    final_drive_ratio,
    driveline_efficiency,
    vehicle_mass,
    full_load_speeds,
    full_load_torques,
):
    """
    Speed and acceleration points per gear are calculated based on
    full load curve, new version works with array and
    forbid acceleration over the maximum vehicle speed.

    :param gear_box_ratios:
        Gear box ratio.
    :type gear_box_ratios: list

    :param idle_engine_speed:
        Idle engine speed.
    :type idle_engine_speed: tuple

    :param tyre_radius:
        Tyre radius.
    :type tyre_radius: float

    :param driveline_slippage:
        Drive line slippage.
    :type driveline_slippage: int

    :param final_drive_ratio:
        Final drive.
    :type final_drive_ratio: float

    :param driveline_efficiency:
        Driveline efficiency.
    :type driveline_efficiency: float

    :param vehicle_mass:
        Vehicle mass.
    :type vehicle_mass: float

    :param full_load_speeds:
        Full load speeds.
    :type full_load_speeds: numpy.array

    :param full_load_torques:
    :type full_load_torques: numpy.array

    :return: speed_per_gear, acc_per_gear
    """

    speed_per_gear, acc_per_gear = [], []

    full_load_speeds = np.array(full_load_speeds)
    full_load_torque = np.array(full_load_torques)

    for j in range(len(gear_box_ratios)):
        mask = full_load_speeds > 1.25 * idle_engine_speed[0]

        temp_speed = (
            2
            * np.pi
            * tyre_radius
            * full_load_speeds[mask]
            * (1 - driveline_slippage)
            / (60 * final_drive_ratio * gear_box_ratios[j])
        )
        speed_per_gear.append(temp_speed)

        temp_acc = (
            full_load_torque[mask]
            * (final_drive_ratio * gear_box_ratios[j])
            * driveline_efficiency
            / (tyre_radius * vehicle_mass)
        )

        acc_per_gear.append(temp_acc)

    return speed_per_gear, acc_per_gear


dsp.add_data("degree", 2)


@sh.add_function(dsp, outputs=["coefs_per_gear"])
def get_tan_coefs(speed_per_gear, acc_per_gear, degree):
    """
    Calculate the coefficients of the polynomial for each gear
    Full load curve is fitted to a polynomial of degree.

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: list[tuple[float]]

    :param acc_per_gear:
        Acceleration per gear.
    :type acc_per_gear: numpy.array

    :param degree:
        Degree.
    :type degree: int

    :return: coefs_per_gear:
        The coefficients of the polynomial for each gear.
    :rtype: list[tuple[float]]]
    """
    it = zip(speed_per_gear, acc_per_gear)
    return [np.polyfit(s, a, degree) for s, a in it]


dsp.add_function(
    function=sh.bypass, inputs=["motor_max_power"], outputs=["engine_max_power"]
)


@sh.add_function(dsp, outputs=["poly_spline", "start", "stop"])
def ev_curve(
    fuel_type,
    engine_max_power,
    tyre_radius,
    motor_max_torque,
    final_drive_ratio,
    driveline_efficiency,
    vehicle_mass,
    vehicle_max_speed,
):
    """
    Full load curve of EVs (based on Yinglong).

    :param fuel_type:
        Fuel type.
    :type fuel_type: str

    :param engine_max_power:
        Engine maximum power.
    :type engine_max_power: float

    :param tyre_radius:
        Tyre radius.[m]
    :type tyre_radius: float

    :param motor_max_torque:
        Motor maximum torque.
    :type motor_max_torque: float

    :param final_drive_ratio:
        Final drive
    :type final_drive_ratio: float

    :param driveline_efficiency:
        Drive line efficiency.
    :type driveline_efficiency: float

    :param vehicle_mass:
        Vehicle mass.
    :type vehicle_mass: float

    :param vehicle_max_speed:
        Vehicle maximum speed. [m/s]
    :type vehicle_max_speed: int

    :return:
        Acceleration potential curves of Electric Vehicle
    :rtype: list[tuple[float]]]
    """
    if fuel_type != "electricity":
        return [sh.NONE] * 3
    from scipy.interpolate import CubicSpline

    eff, fdr = driveline_efficiency, final_drive_ratio

    max_a = motor_max_torque * fdr * eff / (tyre_radius * vehicle_mass)  # m/s2
    s = np.arange(0, vehicle_max_speed + 0.1, 0.1)  # m/s
    a = np.round((s[0]), 2) - 0.01
    b = np.round((s[-1]), 2) + 0.01
    prefix = np.asarray([a - k * 0.1 for k in range(10, -1, -1)])
    suffix = np.asarray([b + k * 0.1 for k in range(0, 11, 1)])

    with np.errstate(divide="ignore"):
        a = (engine_max_power * 1e3 * eff / (s * vehicle_mass)).clip(0, max_a)

    return (
        [
            CubicSpline(
                np.append(np.append(prefix, s), suffix),
                np.append(
                    np.append(np.repeat(a[0], len(prefix)), a),
                    np.repeat(a[-1], len(suffix)),
                ),
            )
        ],
        [s[0]],
        [s[-1]],
    )


@sh.add_function(dsp, inputs_kwargs=True, outputs=["poly_spline"])
def get_cubic_splines_of_speed_acceleration_relationship(
    speed_per_gear, acc_per_gear, use_cubic=True
):
    """
    Based on speed/acceleration points per gear, cubic splines are calculated
    (old MFC).

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: list

    :param acc_per_gear:
        Acceleration per gear.
    :type acc_per_gear: list

    :param use_cubic:
        Use cubic.
    :type use_cubic: bool

    :return:
        Engine acceleration potential curves.
    :rtype: list[tuple[float]]]
    """
    if not use_cubic:
        return sh.NONE
    from scipy.interpolate import CubicSpline as Spl

    v, a = np.asarray(speed_per_gear), np.asarray(acc_per_gear)
    v = (
        np.round(v[:, 0, None], 2) - 0.01 - np.linspace(0, 1, 11)[::-1],
        v,
        np.round(v[:, -1, None], 2) + 0.01 + np.linspace(0, 1, 11),
    )
    a = np.tile(a[:, 0, None], 11), a, np.tile(a[:, -1, None], 11)
    return [Spl(*d) for d in zip(np.concatenate(v, 1), np.concatenate(a, 1))]


@sh.add_function(dsp, inputs_kwargs=True, outputs=["poly_spline"])
def get_spline_out_of_coefs(coefs_per_gear, speed_per_gear, use_cubic=False):
    """
    Use the coefficients to get a "spline" that could be used.
    AT TIME IT IS USED AS calculate_curve_to_use FUNCTION IS USING SPLINES.

    :param coefs_per_gear:
        Coefficients per gear.
    :type coefs_per_gear: list

    :param speed_per_gear:
        Starting speed.
    :type speed_per_gear: float

    :param use_cubic:
        Use cubic.
    :type use_cubic: bool

    :return:
        Poly spline functions.
    :rtype: list[tuple[float]]
    """
    if use_cubic:
        return sh.NONE
    from scipy.interpolate import interp1d

    degree = len(coefs_per_gear[0]) - 1
    vars_ = np.arange(degree, -1, -1)

    spline_from_poly = []

    # For the first gear, some points are added at the beginning to avoid
    # unrealistic drops
    x_new = np.insert(
        np.arange(speed_per_gear[0][0], 70, 0.1), [0, 0], [0, speed_per_gear[0][0] / 2]
    )
    a_new = np.array([np.dot(coefs_per_gear[0], np.power(i, vars_)) for i in x_new])
    a_new[0] = a_new[2]
    a_new[1] = a_new[2]
    spline_from_poly.append(interp1d(x_new, a_new, fill_value="extrapolate"))

    for fit_coef in coefs_per_gear[1:]:
        x_new = np.arange(0, 70, 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars_)) for i in x_new])
        spline_from_poly.append(interp1d(x_new, a_new, fill_value="extrapolate"))

    return spline_from_poly


@sh.add_function(dsp, outputs=["discrete_poly_spline"])
def define_discrete_poly(poly_spline, sp_bins):
    """
    Define discrete poly.

    :param poly_spline:
        Poly spline.
    :type poly_spline: list[tuple[float]]]

    :param sp_bins:
        Speed bins.
    :type sp_bins: numpy.array

    :rtype: list[tuple[float]]]
    """
    return [acc(sp_bins) for acc in poly_spline]


# Start/stop speed for each gear
@sh.add_function(
    dsp, outputs=["start", "stop", "speed_per_gear_updated", "acc_per_gear_updated"]
)
def get_start_stop(
    gear_box_ratios, acc_per_gear, vehicle_max_speed, speed_per_gear, poly_spline
):
    """
    Calculate Speed boundaries for each gear.

    :param gear_box_ratios:
        Gear box ratios.
    :type gear_box_ratios: list

    :param acc_per_gear:
        Acceleration per gear
    :type acc_per_gear: numpy.array

    :param vehicle_max_speed:
        Vehicle maximum speed.
    :type vehicle_max_speed: int

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: numpy.array

    :param poly_spline:
        Poly spline.
    :type poly_spline: list

    :return:
        Start and Stop for each gear.
    :rtype: numpy.array, numpy.array
    """
    # To ensure that a higher gear starts from higher speed
    for j in range(len(gear_box_ratios) - 1, 0, -1):
        for k in range(len(speed_per_gear[j])):
            if speed_per_gear[j - 1][0] < speed_per_gear[j][0]:
                break
            else:
                # If the gear ratios are not declining,
                # there is an error in the database. Return error.
                return
                # speed_per_gear[j] = speed_per_gear[j][3:]

    # Find where the curve of each gear cuts the next one.
    for j in range(len(gear_box_ratios) - 1):
        for k in range(np.minimum(len(speed_per_gear[j]), len(speed_per_gear[j + 1]))):
            if (speed_per_gear[j][k] > speed_per_gear[j + 1][0]) & (
                poly_spline[j + 1](speed_per_gear[j][k])
                > poly_spline[j](speed_per_gear[j][k])
            ):
                max_point = k
                speed_per_gear[j] = speed_per_gear[j][:max_point]
                acc_per_gear[j] = acc_per_gear[j][:max_point]
                break

    # The limits of the gears that should be provided to the gear shifting model
    start = []
    stop = []
    for i in speed_per_gear:
        start.append(i[0])
        stop.append(min(i[-1], vehicle_max_speed))
    start[0] = 0
    return start, stop, speed_per_gear, acc_per_gear


@sh.add_function(dsp, outputs=["sp_bins"])
def define_sp_bins(stop):
    """
    Define speed bins.

    :param stop:
        Stop speed per gear curve.
    :type stop: list

    :return:
        Speed bins.
    :rtype: list[float]
    """
    return np.arange(0, stop[-1] + 0.1, 0.1)


@sh.add_function(dsp, outputs=["discrete_car_res_curve_force"])
def define_discrete_car_res_curve_force(car_res_curve_force, sp_bins):
    """
    Define discrete resistance force.

    :param car_res_curve_force:
        Resistance force.
    :type car_res_curve_force

    :param sp_bins:
        Speed boundaries.
    :type sp_bins: numpy.array

    :return:
        Discrete resistance force.
    :rtype: numpy.array
    """
    return car_res_curve_force(sp_bins)


@sh.add_function(dsp, outputs=["discrete_car_res_curve"])
def define_discrete_car_res_curve(car_res_curve, sp_bins):
    """
    Define discrete car resistance curve.

    :param car_res_curve:
        Car resistance curve.
    :type car_res_curve: numpy.array[tuple[float]]

    :param sp_bins:
        Speed bins.
    :type sp_bins: numpy.array[float]

    :return:
        Discrete car resistance curve
    :rtype: numpy.array[float]
    """
    return car_res_curve(sp_bins)


# Calculates a spline with the resistances when f0, f1, f2 input by the user
@sh.add_function(dsp, outputs=["f0", "f1", "f2"])
def calculate_veh_road_loads(vehicle_mass, type_of_car, car_width, car_height):
    from .co2mpas import estimate_f_coefficients

    return estimate_f_coefficients(vehicle_mass, type_of_car, car_width, car_height)


dsp.add_data("angle_slopes", 0.0)


# Calculates a spline with the resistances when f0, f1, f2 input by the user
@sh.add_function(dsp, outputs=["car_res_curve", "car_res_curve_force"])
def get_resistances(f0, f1, f2, sp_bins, vehicle_mass, angle_slopes=0.0, g=9.81):
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

    :param vehicle_mass:
        Vehicle mass.
    :type vehicle_mass: float

    :param angle_slopes:
        Angle slope of the road.
    :type angle_slopes: float

    :param g:
        Acceleration due to gravity.
    :type g: float

    :param use_estimated_res:
        Use estimated resistances.
    :type use_estimated_res: bool

    :return: resistance_spline_curve, resistance_spline_curve_f
        Resistance forces being applied per speed.
    :rtype: scipy.interpolate._cubic.CubicSpline, scipy.interpolate._cubic.CubicSpline
    """
    from .co2mpas import veh_resistances

    return veh_resistances(f0, f1, f2, sp_bins, vehicle_mass, angle_slopes, g)


# The maximum force that the vehicle can have on the road
@sh.add_function(dsp, outputs=["Alimit"])
def Armax(car_type, vehicle_mass, engine_max_power, road_type=1):
    """
    Calculate the maximum acceleration possible for the vehicle object my_car,
    under road_type conditions.

    :param car_type:
        Car type.
    :type car_type: int

    :param vehicle_mass:
        Vehicle mass.
    :type vehicle_mass: float

    :param engine_max_power:
        Maximum engine power.
    :type engine_max_power: float

    :param road_type: road condition (1: normal, 2: wet, 3: icy)
        Road type.
    :type road_type: int

    :return:
        Vehicle maximum acceleration.
    :rtype: float
    """

    mass = {2: 0.6, 4: 0.45}.get(car_type, 1) * vehicle_mass  # Load distribution.
    mh_base = {1: 0.75, 2: 0.25}.get(road_type, 0.1)  # Friction coeff.

    alpha, beta = 43.398, 5.1549
    mh = mh_base * (alpha * np.log(engine_max_power) + beta) / 190
    # * cos(f) for the gradient of the road. Here we consider as 0

    return mass * 9.8066 * mh / vehicle_mass


@sh.add_function(dsp, outputs=["curves"])
def calculate_curves_to_use(poly_spline, start, stop, Alimit, car_res_curve, sp_bins):
    """
    Calculate the final speed acceleration curves based on full load curves and
    resistances for all curves.

    :param poly_spline:
        Poly spline.
    :type poly_spline:

    :param start:
        Start speed per gear.
    :type start: list

    :param stop:
        Stop speed per gear.
    :type stop: list

    :param Alimit:
        Maximum acceleration possible.
    :type Alimit: float

    :param car_res_curve:
        Car resistance curve.
    :type car_res_curve:

    :param sp_bins:
        Speed boundaries per gear.
    :type sp_bins: numpy.array

    :return:
        Final speed and acceleration curves.
    :rtype: list
    """
    from scipy.interpolate import interp1d

    res = []

    for gear, acc in enumerate(poly_spline):
        start_ = start[gear] * 0.9
        stop_ = stop[gear] + 0.1
        if len(poly_spline) == 1:  # for EV
            stop_ = stop[gear]

        final_acc = acc(sp_bins) - car_res_curve(sp_bins)
        final_acc[final_acc > Alimit] = Alimit

        final_acc[(sp_bins < start_)] = 0
        final_acc[(sp_bins > stop_)] = 0
        final_acc[final_acc < 0] = 0

        res.append(interp1d(sp_bins, final_acc))

    return res


@sh.add_function(dsp, outputs=["curves_dec"])
def calculate_deceleration_curves_to_use(stop):
    """
    Calculate deceleration curves .

    :param stop:
        Stop speed per gear.
    :type stop: list

    :return:
        Deceleration curves.
    :rtype: list
    """
    from scipy.interpolate import interp1d

    ppar = [0.0045, -0.1710, -1.8835]
    dec_curves = np.poly1d(ppar)

    curves_dec = []
    for gear in range(len(stop)):
        sp_bins = np.arange(0, stop[gear] + 0.1, 0.1)
        final_dec = []
        for k in range(len(sp_bins)):
            final_dec.append(min(dec_curves(sp_bins[k]), -1))
        curves_dec.append(interp1d(sp_bins, final_dec))

    return curves_dec


@sh.add_function(dsp, outputs=["discrete_acceleration_curves"])
def define_discrete_acceleration_curves(curves, start, stop):
    """
    Define discrete acceleration curves.

    :param curves:
        Curves
    :type curves:

    :param start:
        Start speed per gear.
    :type start: list

    :param stop:
        Stop speed per gear.
    :type stop: list

    :rtype: list[dict[numpy.array[float]]]
    """
    res = []
    for gear, f in enumerate(curves):
        x = np.arange(start[gear], stop[gear], 0.2)
        res.append(dict(x=x, y=f(x)))
    return res


@sh.add_function(dsp, outputs=["discrete_deceleration_curves"])
def define_discrete_deceleration_curves(curves_dec, start, stop):
    """
    Define discrete deceleration curves.

    :param curves_dec:
        Deceleration curves.
    :type curves_dec: list

    :param start:
        Start speed per gear.
    :type start: list

    :param stop:
        Stop speed per gear.
    :type stop: list

    :rtype: list[dict[numpy.array[float]]]
    """
    res = []
    for gear, f in enumerate(curves_dec):
        x = np.arange(start[gear], stop[gear], 0.2)
        res.append(dict(x=x, y=f(x)))
    return res


# Extract speed acceleration Splines
@sh.add_function(dsp, inputs_kwargs=True, inputs_defaults=True, outputs=["gs"])
def gear_linear(speed_per_gear_updated, gear_shifting_style, use_linear_gs=True):
    """
    Return the gear limits based on gear_shifting_style, using linear gear
    swifting strategy.

    :param speed_per_gear_updated:
        Updated speed per gear based on where the curve each gear cuts the next
        one.
    :type speed_per_gear_updated: numpy.array[list[float]]

    :param gear_shifting_style:
        Gear shifting style.
    :type gear_shifting_style: float

    :param use_linear_gs:
        Use linear gear shifting.
    :type use_linear_gs: bool

    :return:
        Gear limits.
    :rtype: list
    """
    if not use_linear_gs:
        return sh.NONE
    n_gears = len(speed_per_gear_updated)

    gear_shifting_style = min(gear_shifting_style, 1)
    gear_shifting_style = max(gear_shifting_style, 0)

    gs = []

    for gear in range(n_gears - 1):
        speed_by_gs = speed_per_gear_updated[gear][
            -1
        ] * gear_shifting_style + speed_per_gear_updated[gear][0] * (
            1 - gear_shifting_style
        )
        speed_for_continuity = speed_per_gear_updated[gear + 1][0]
        cutoff_s = max(speed_by_gs, speed_for_continuity)

        gs.append(cutoff_s)

    return gs


dsp.add_function(
    function_id="define_idle_engine_speed",
    function=sh.bypass,
    inputs=["idle_engine_speed_median", "idle_engine_speed_std"],
    outputs=["idle_engine_speed"],
)


@sh.add_function(dsp, outputs=["tans"])
def find_list_of_tans_from_coefs(coefs_per_gear, start, stop):
    """
    Get coefficients and speed boundaries and return Tans value for per speed
    per gear.

    :param coefs_per_gear:
        Coefficients per gear.
    :type coefs_per_gear: list

    :param start:
        Start speed per gear.
    :type start: list

    :param stop:
        Stop speed per gear.
    :type stop: list

    :return:
        Tangential values (derivative of force of each gear with respect to the
        speed).
    :rtype: list
    """
    degree = len(coefs_per_gear[0]) - 1
    _vars = np.arange(degree, -1, -1)

    tans = []

    for gear, coefs in enumerate(coefs_per_gear):
        x_new = np.arange(start[gear], stop[gear], 0.1)
        a_new = np.array([np.dot(coefs, np.power(i, _vars)) for i in x_new])

        tans.append(np.diff(a_new) * 10)

    return tans


def _find_gs_cut_tans(tmp_min, tan, tmp_min_next, gear_shifting_style):
    """

    Find where gear is changed, based on tans and gear_shifting_style

    :param tmp_min:
        Temporary minimum speed per gear.
    :type tmp_min: int

    :param tan:
        Tangential values.
    :type tan: numpy.array

    :param tmp_min_next:
        The next minimum speed per gear.
    :type tmp_min_next: float

    :param gear_shifting_style:
        Gear shifting style.
    :type gear_shifting_style: float

    :return:
        Gear changing point.
    :rtype: float
    """
    max_tan = np.max(tan)
    min_tan = np.min(tan)
    acc_range = max_tan - min_tan

    # tan starts from positive and goes negative, so I use (1 - cutoff)
    # for the percentage
    if gear_shifting_style > 0.99:
        gear_shifting_style = 1
    elif gear_shifting_style < 0.01:
        gear_shifting_style = 0.01
    tan_cutoff = (1 - gear_shifting_style) * acc_range + min_tan

    # Search_from = int(tmp_min_next * 10)
    search_from = int((tmp_min_next - tmp_min) * 10) + 1

    i_cut = len(tan) - 1
    while tan[i_cut] < tan_cutoff and i_cut >= search_from:
        i_cut -= 1

    gear_cut = tmp_min + i_cut / 10 + 0.1

    return gear_cut


@sh.add_function(dsp, outputs=["gs"])
def default_gs(fuel_type):
    if fuel_type == "electricity":
        return []
    return sh.NONE


@sh.add_function(dsp, inputs_kwargs=True, outputs=["gs"])
def gear_points_from_tan(tans, gear_shifting_style, start, use_linear_gs=False):
    """
    Get the gear cuts based on gear shifting style and tangent values.

    :param tans:
        Tangent values per gear.
    :type tans: list[numpy.array[float]]

    :param gear_shifting_style:
        Gear shifting style.
    :type gear_shifting_style: float

    :param start:
        Start speed per gear curve.
    :type start: list

    :param use_linear_gs:
        Use gear linear to calculate gs.
    :type use_linear_gs: bool

    :return:
        Gear limits
    :rtype: list[float]
    """
    if use_linear_gs:
        return sh.NONE
    n_gears = len(tans)
    gs_cut = [gear_shifting_style for i in range(n_gears)]

    gs = []

    for i in range(n_gears - 1):
        tmp_min = start[i]
        # tmp_max = stop[i]
        tan = tans[i]
        tmp_min_next = start[i + 1]
        cutoff_s = _find_gs_cut_tans(tmp_min, tan, tmp_min_next, gs_cut[i])

        gs.append(cutoff_s)

    return gs


dsp.add_data("sim_start", 0)


@sh.add_function(dsp, outputs=["times"])
def define_times(sim_start, duration, sim_step):
    """
    Define times for simulation.

    :param sim_start:
        Simulation starting time. [s]
    :type sim_start: int

    :param duration:
        Duration of the simulation. [s]
    :type duration: int

    :param sim_step:
        Simulation step. [s]
    :type sim_step: float

    :return:
        Time series.
    :rtype: numpy.array
    """
    return np.arange(sim_start, duration + sim_step, sim_step)


@sh.add_function(dsp, outputs=["driver_simulation_model"])
def define_driver_simulation_model(
    vehicle_mass,
    r_dynamic,
    car_type,
    final_drive_ratio,
    gear_box_ratios,
    gearbox_type,
    engine_max_torque,
    fuel_eng_capacity,
    max_power,
    fuel_engine_stroke,
    fuel_type,
    fuel_turbo,
    type_of_car,
    car_width,
    car_height,
    transmission,
    gs,
    curves,
    curves_dec,
    driver_style,
):
    """
        Defines the drivers simulation model.

    :param vehicle_mass:
        Vehicle mass.
    :type: float

    :param r_dynamic:
        Dynamic radius.
    :type: float

    :param car_type:
        Car type.
    :type: int

    :param final_drive_ratio:
        Final drive ratio.
    :type: float

    :param gear_box_ratios:
        Gear box ratios.
    :type: list

    :param gearbox_type:
        Gearbox type.
    :type: str

    :param engine_max_torque:
        Engine maximum torque.
    :type: float

    :param fuel_eng_capacity:
        Fuel engine capacity.
    :type: float

    :param max_power:
        Maximum power.
    :type: int

    :param fuel_engine_stroke:
        Fuel engine stroke.
    :type: float

    :param fuel_type:
        Fuel type.
    :type: str

    :param fuel_turbo:
        Fuel turbo.
    :type: str

    :param type_of_car:
        Type of car.
    :type: str

    :param car_width:
        Car width.
    :type: float

    :param car_height:
        Car height.
    :type: float

    :param transmission:
        Vehicle transmission system.
    :type: str

    :param gs:
        Gear cuts.
    :type: list

    :param curves:
        Acceleration potential curves per gear.
    :type: list

    :param curves_dec:
        Deceleration potential curves per gear.
    :type: list

    :param driver_style:
        Driver style.
    :type: int

    :return:
        Driver simulation model.
    :rtype
    """
    from .driver import Driver

    if fuel_type == "electricity":
        return sh.NONE
    return Driver(
        vehicle_mass,
        car_type,
        final_drive_ratio,
        gearbox_type,
        max_power,
        fuel_type,
        type_of_car,
        car_width,
        car_height,
        transmission,
        gs,
        curves,
        curves_dec,
        driver_style,
        r_dynamic,
        gear_box_ratios,
        engine_max_torque,
        fuel_eng_capacity,
        fuel_engine_stroke,
        fuel_turbo,
    )


@sh.add_function(dsp, outputs=["driver_simulation_model"])
def define_ev_driver_simulation_model(
    vehicle_mass,
    car_type,
    final_drive_ratio,
    gearbox_type,
    max_power,
    fuel_type,
    type_of_car,
    car_width,
    car_height,
    transmission,
    gs,
    curves,
    curves_dec,
    driver_style,
):
    """
        Defines the drivers simulation model.

    :param vehicle_mass:
        Vehicle mass.
    :type: float

    :param car_type:
        Car type.
    :type: int

    :param final_drive_ratio:
        Final drive ratio.
    :type: float

    :param gearbox_type:
        Gearbox type.
    :type: str

    :param max_power:
        Maximum power.
    :type: int

    :param fuel_type:
        Fuel type.
    :type: str

    :param car_width:
        Car width.
    :type: float

    :param car_height:
        Car height.
    :type: float

    :param type_of_car:
        Type of car.
    :type: str

    :param transmission:
        Vehicle transmission system.
    :type: str

    :param gs:
        Gear cuts.
    :type: list

    :param curves:
        Acceleration potential curves per gear.
    :type: list

    :param curves_dec:
        Deceleration potential curves per gear.
    :type: list

    :param driver_style:
        Driver style.
    :type: int

    :return:
        Driver simulation model.
    :rtype
    """
    from .driver import Driver

    if fuel_type != "electricity":
        return sh.NONE
    return Driver(
        vehicle_mass,
        car_type,
        final_drive_ratio,
        gearbox_type,
        max_power,
        fuel_type,
        type_of_car,
        car_width,
        car_height,
        transmission,
        gs,
        curves,
        curves_dec,
        driver_style,
    )


@sh.add_function(dsp, outputs=["gears", "gear_counts", "velocities", "accelerations"])
def run_simulation(
    driver_simulation_model, starting_velocity, sim_step, times, desired_velocity
):
    """
    Run simulation.

    :param driver_simulation_model:
        Driver simulation model.
    :type driver_simulation_model:

    :param starting_velocity:
        Current speed.
    :type starting_velocity: int

    :param sim_step:
        Simulation step.
    :type sim_step: int (sec)

    :param times:
        Sample time series.
    :type times: np.array

    :param desired_velocity:
        Desired velocity.
    :type desired_velocity: int

    :return:
        Gears & velocities.
    :rtype: list[numpy.array]
    """
    model = driver_simulation_model.reset(starting_velocity)
    r = [(model._gear, model._gear_count, starting_velocity, 0)]  # Gather data
    r.extend(model(sim_step, desired_velocity) for t in times)
    return list(zip(*r))[:4]


@sh.add_function(dsp, outputs=["fp"])
def light_co2mpas_series(
    vehicle_mass,
    r_dynamic,
    car_type,
    final_drive_ratio,
    gear_box_ratios,
    gearbox_type,
    type_of_car,
    car_width,
    car_height,
    engine_max_torque,
    fuel_eng_capacity,
    max_power,
    fuel_engine_stroke,
    fuel_type,
    fuel_turbo,
    sim_step,
    velocities,
    gs,
    **kwargs
):
    """
        Computes the CO2 emissions in grams for
        a series of speed profile.

    :param vehicle_mass:
        Vehicle mass.
    :type: float

    :param r_dynamic:
        Dynamic radius.
    :type: float

    :param car_type:
        Car type.
    :type: int

    :param final_drive_ratio:
        Final drive ratio.
    :type: float

    :param gear_box_ratios:
        Gear box ratios.
    :type: list

    :param gearbox_type:
        Gearbox type.
    :type: str

    :param type_of_car:
        Type of car.
    :type: str

    :param car_width:
        Car width.
    :type: float

    :param car_height:
        Car height.
    :type: float

    :param engine_max_torque:
        Engine maximum torque.
    :type: float

    :param fuel_eng_capacity:
        Fuel engine capacity.
    :type: float

    :param max_power:
        Maximum power.
    :type: int

    :param fuel_engine_stroke:
        Fuel engine stroke.
    :type: float

    :param fuel_type:
        Fuel type.
    :type: str

    :param fuel_turbo:
        Fuel turbo.
    :type: str

    :param sim_step:
        Simulation step.
    :type sim_step: int (sec)

    :param velocities:
    :param gs:
        Gear cuts.
    :type: list

    :param kwargs:
        Key word arguments.
    :type: dict

    :return:
    """

    from co2mpas_driver.common import vehicle_specs_class as vcc, gear_functions as fg
    from .co2mpas import estimate_f_coefficients

    gear_list = {}
    clutch_list = []
    gear_list_flag = False
    if "gear_list" in kwargs:
        gear_list_flag = True
        gear_list = kwargs["gear_list"]
        if "clutch_duration" in kwargs:
            clutch_duration = kwargs["clutch_duration"]
        else:
            clutch_duration = int(0.5 % sim_step)
        clutch_list = fg.create_clutch_list(gear_list, clutch_duration)

    hardcoded_params = vcc.hardcoded_params()

    # n_wheel_drive = my_car.car_type
    road_loads = estimate_f_coefficients(
        vehicle_mass, type_of_car, car_width, car_height, passengers=0
    )

    slope = 0
    # FIX First convert km/h to m/s in order to have acceleration in m/s^2
    ap = np.diff([i / (3.6 * sim_step) for i in velocities])

    # gear number and gear count for shifting duration
    # simulated_gear = [0, 30]
    fp = []

    if gearbox_type == "manual":
        veh_params = hardcoded_params.params_gearbox_losses["Manual"]
        gb_type = 0
    else:
        veh_params = hardcoded_params.params_gearbox_losses["Automatic"]
        gb_type = 1

    # gear is the current gear and gear_count counts the time-steps
    # in order to prevent continuous gear shifting.
    gear = 0
    # Initializing gear count.
    gear_count = 30

    for i in range(1, len(velocities)):
        speed = velocities[i]
        acceleration = ap[i - 1]

        if gear_list_flag:
            gear = gear_list[i]
            gear_count = clutch_list[i]
        else:
            gear, gear_count = fg.gear_for_speed_profiles(
                gs, speed / 3.6, gear, gear_count, gb_type
            )
        from co2mpas_driver.common.generic_co2mpas import light_co2mpas_instant

        fc = light_co2mpas_instant(
            vehicle_mass,
            r_dynamic,
            car_type,
            final_drive_ratio,
            gear_box_ratios,
            veh_params,
            engine_max_torque,
            fuel_eng_capacity,
            speed,
            acceleration,
            max_power,
            fuel_engine_stroke,
            fuel_type,
            fuel_turbo,
            hardcoded_params,
            road_loads,
            slope,
            gear,
            gear_count,
            sim_step,
        )

        fp.append(fc)

    return fp


if __name__ == "__main__":
    dsp.plot()
