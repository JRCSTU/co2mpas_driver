# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to processes a CO2MPAS input file.

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
from co2mpas_driver.model.co2mpas import get_full_load, \
    calculate_full_load_speeds_and_powers, calculate_full_load_torques

dsp = sh.Dispatcher(name='model')
dsp.add_func(get_full_load, outputs=['full_load_curve'])
dsp.add_func(
    calculate_full_load_speeds_and_powers,
    outputs=['full_load_speeds', 'full_load_powers']
)
dsp.add_func(
    calculate_full_load_torques,
    outputs=['full_load_torques']
)


# Speed and acceleration ranges and points for each gear
@sh.add_function(dsp, outputs=['speed_per_gear', 'acc_per_gear'])
def get_speeds_n_accelerations_per_gear(
        gear_box_ratios, idle_engine_speed, tyre_radius, driveline_slippage,
        final_drive_ratio, driveline_efficiency, vehicle_mass, full_load_speeds,
        full_load_torques):
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
    :type full_load_speeds: ndarray

    :param full_load_torques:
    :type full_load_torques: ndarray

    :return: speed_per_gear
    """

    fls, flt = np.asarray(full_load_speeds), np.asarray(full_load_torques)
    gbr, b = np.asarray(gear_box_ratios), fls > 1.25 * idle_engine_speed[0]

    cv = 2 * np.pi * tyre_radius * (1 - driveline_slippage) / (60 * final_drive_ratio)
    ca = final_drive_ratio * driveline_efficiency / (tyre_radius * vehicle_mass)
    return cv * fls[None, b] / gbr[:, None], ca * gbr[:, None] * flt[None, b]


dsp.add_data('degree', 4)


@sh.add_function(dsp, outputs=['coefs_per_gear'])
def get_tan_coefs(speed_per_gear, acc_per_gear, degree):
    """
    Full load curve is fitted to a polynomial of degree.

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: numpy.array

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


@sh.add_function(dsp, outputs=['poly_spline', 'Start', 'Stop'])
def ev_curve(fuel_type, engine_max_power, tyre_radius,
             motor_max_torque, final_drive_ratio,
             driveline_efficiency, vehicle_mass, vehicle_max_speed):
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
        Aceeleration potential curves of Electric Vehicle
    :rtype: list
    """
    if fuel_type != 'electricity':
        return [sh.NONE] * 3
    from scipy.interpolate import CubicSpline

    veh_max_acc = (motor_max_torque * final_drive_ratio) * driveline_efficiency \
                  / (tyre_radius * vehicle_mass)  # m/s2

    speeds = np.arange(0, vehicle_max_speed + 0.1, 0.1)  # m/s

    with np.errstate(divide='ignore'):
        accelerations = engine_max_power * 1000 * \
                        driveline_efficiency / (speeds * vehicle_mass)

    accelerations[accelerations > veh_max_acc] = veh_max_acc
    accelerations[accelerations < 0] = 0

    cs_acc_ev = CubicSpline(speeds, accelerations)
    start = 0
    stop = vehicle_max_speed

    return [cs_acc_ev], [start], [stop]


@sh.add_function(dsp, inputs_kwargs=True, outputs=['poly_spline'])
def get_cubic_splines_of_speed_acceleration_relationship(
        speed_per_gear, acc_per_gear, use_cubic=True):
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
    :rtype: list
    """
    if not use_cubic:
        return sh.NONE
    from scipy.interpolate import CubicSpline
    cs_acc_per_gear = []
    for j in range(len(speed_per_gear)):
        # cs_acc_per_gear.append([])
        a = np.round((speed_per_gear[j][0]), 2) - 0.01
        b = np.round((speed_per_gear[j][-1]), 2) + 0.01
        prefix_list = [a - k * 0.1 for k in range(10, -1, -1)]
        suffix_list = [b + k * 0.1 for k in range(0, 11, 1)]
        cs_acc_per_gear.append(CubicSpline(
            prefix_list + list(speed_per_gear[j]) + suffix_list,
            [acc_per_gear[j][0]] * len(prefix_list) + list(acc_per_gear[j]) + [
                acc_per_gear[j][-1]] * len(suffix_list))
        )

    return cs_acc_per_gear


@sh.add_function(dsp, inputs_kwargs=True, inputs_defaults=True,
                 outputs=['poly_spline'])
def get_spline_out_of_coefs(coefs_per_gear, starting_speed, use_cubic=False):
    """
    Use the coefficients to get a "spline" that could be used.
    AT TIME IT IS USED AS calculate_curve_to_use FUNCTION IS USING SPLINES.

    :param coefs_per_gear:
        Coefficients per gear.
    :type coefs_per_gear: list

    :param starting_speed:
        Starting speed.
    :type starting_speed: float

    :param use_cubic:
        Use cubic.
    :type use_cubic: bool

    :return:
        Poly spline functions.
    :rtype: list
    """
    if use_cubic:
        return sh.NONE
    from scipy.interpolate import interp1d
    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    poly_spline = []

    """
    For the first gear, some points are added at the beginning to avoid
    unrealistic drops 
    """
    x_new = np.insert(np.arange(starting_speed, 70, 0.1), [0, 0],
                      [0, starting_speed / 2])
    a_new = np.array(
        [np.dot(coefs_per_gear[0], np.power(i, vars)) for i in x_new])
    a_new[0] = a_new[2]
    a_new[1] = a_new[2]
    poly_spline.append(interp1d(x_new, a_new, fill_value='extrapolate'))

    for fit_coef in coefs_per_gear[1:]:
        x_new = np.arange(0, 70, 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars)) for i in x_new])
        poly_spline.append(interp1d(x_new, a_new, fill_value='extrapolate'))

    return poly_spline


@sh.add_function(dsp, outputs=['discrete_poly_spline'])
def define_discrete_poly(poly_spline, sp_bins):
    return [acc(sp_bins) for acc in poly_spline]


# Start/stop speed for each gear
@sh.add_function(dsp, outputs=['Start', 'Stop'])
def get_start_stop(gear_box_ratios, vehicle_max_speed, speed_per_gear,
                   acc_per_gear,
                   poly_spline):
    """
    Calculate Speed boundaries for each gear.

    :param gear_box_ratios:
        Gear box ratios.
    :type gear_box_ratios: list

    :param vehicle_max_speed:
        Vehicle maximum speed.
    :type vehicle_max_speed: int

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: numpy.array

    :param acc_per_gear:
        Acceleration per gear.
    :type acc_per_gear: numpy.array

    :param poly_spline:
        Poly spline.
    :type poly_spline: list

    :return:
        Start and Stop for each gear.
    :rtype: list, list
    """
    speed_per_gear = np.array(speed_per_gear).tolist()
    acc_per_gear = np.array(acc_per_gear).tolist()
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
    for j in range(len(speed_per_gear) - 1):
        for k in range(
                np.minimum(len(speed_per_gear[j]), len(speed_per_gear[j + 1]))):
            if (speed_per_gear[j][k] > speed_per_gear[j + 1][0]) & (
                    poly_spline[j + 1](speed_per_gear[j][k]) >
                    poly_spline[j](speed_per_gear[j][k])):
                max_point = k
                speed_per_gear[j] = speed_per_gear[j][:max_point]
                acc_per_gear[j] = acc_per_gear[j][:max_point]
                break

    # The limits of the gears that should be provided to the gear shifting model
    Start = []
    Stop = []
    for i in speed_per_gear:
        Start.append(i[0])
        Stop.append(min(i[-1], vehicle_max_speed))
    Start[0] = 0
    return Start, Stop


# Start/stop speed for each gear
@sh.add_function(dsp, outputs=['sp_bins'])
def define_sp_bins(Stop):
    return np.arange(0, Stop[-1] + 1, 0.01)


@sh.add_function(dsp, outputs=['discrete_car_res_curve_force'])
def define_discrete_car_res_curve_force(car_res_curve_force, sp_bins):
    discrete_car_res_curve_force = car_res_curve_force(sp_bins)
    return discrete_car_res_curve_force


@sh.add_function(dsp, outputs=['discrete_car_res_curve'])
def define_discrete_car_res_curve(car_res_curve, sp_bins):
    discrete_car_res_curve = car_res_curve(sp_bins)
    return discrete_car_res_curve


# Calculate Curves
@sh.add_function(dsp, outputs=['car_res_curve', 'car_res_curve_force'])
def get_resistances(type_of_car, vehicle_mass, car_width, car_height, sp_bins):
    """
    Calculate resistances and return spline.

    :param type_of_car:
        Type of car.
    :type type_of_car: str

    :param vehicle_mass:
        Vehicle mass.
    :type vehicle_mass: float

    :param car_width:
        Car width.
    :type car_width: float

    :param car_height:
        Car height.
    :type car_height: float

    :param sp_bins:
        Speed bins.
    :type sp_bins: numpy.array

    :return:
        Car resistance curve.
    :rtype: CubicSpline
    """

    from .co2mpas import estimate_f_coefficients, veh_resistances, Armax
    f0, f1, f2 = estimate_f_coefficients(vehicle_mass, type_of_car, car_width,
                                         car_height)
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2,
                                                         list(sp_bins),
                                                         vehicle_mass)
    return car_res_curve, car_res_curve_force


# The maximum force that the vehicle can have on the road
@sh.add_function(dsp, outputs=['Alimit'])
def Armax(car_type, vehicle_mass, engine_max_power, road_type=1):
    """
    Calculating the maximum acceleration possible for the vehicle object my_car,
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
    if car_type == 2:  # forward-wheel drive vehicles
        fmass = 0.6 * vehicle_mass
    elif car_type == 4:  # rear-wheel drive vehicles
        fmass = 0.45 * vehicle_mass
    else:  # all-wheel drive vehicles, 4x4
        fmass = 1 * vehicle_mass

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

    return Frmax / vehicle_mass


@sh.add_function(dsp, outputs=['Curves'])
def calculate_curves_to_use(poly_spline, Start, Stop, Alimit, car_res_curve,
                            sp_bins):
    """
    Get the final speed acceleration curves based on full load curves and
    resistances for all curves.

    :param poly_spline:
        Poly spline.
    :type poly_spline:

    :param Start:
        Start speed per gear.
    :type Start: list

    :param Stop:
        Stop speed per gear.
    :type Stop: list

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
    Res = []

    for gear, acc in enumerate(poly_spline):
        start = Start[gear] * 0.9
        stop = Stop[gear] + 0.1

        final_acc = acc(sp_bins) - car_res_curve(sp_bins)
        final_acc[final_acc > Alimit] = Alimit

        final_acc[(sp_bins < start)] = 0
        final_acc[(sp_bins > stop)] = 0
        final_acc[final_acc < 0] = 0

        Res.append(interp1d(sp_bins, final_acc))

    return Res


@sh.add_function(dsp, outputs=['starting_speed'])
def get_starting_speed(speed_per_gear):
    starting_speed = speed_per_gear[0][0]
    return starting_speed


@sh.add_function(dsp, outputs=['discrete_acceleration_curves'])
def define_discrete_acceleration_curves(Curves, Start, Stop):
    res = []
    for gear, f in enumerate(Curves):
        x = np.arange(Start[gear], Stop[gear], 0.2)
        res.append(dict(x=x, y=f(x)))
    return res


# Extract speed acceleration Splines
@sh.add_function(dsp, inputs_kwargs=True, inputs_defaults=True, outputs=['gs'])
def gear_linear(speed_per_gear, gs_style, use_linear_gs=True):
    """
    Return the gear limits based on gs_style, using linear gear swifting
    strategy.

    :param speed_per_gear:
        Speed per gear.
    :type speed_per_gear: numpy.array

    :param gs_style:
        Gear shifting style.
    :type gs_style: float

    :param use_linear_gs:
        Use linear gear shifting.
    :type use_linear_gs: bool

    :return:
        Gear limits.
    :rtype: list
    """
    if not use_linear_gs:
        return sh.NONE
    n_gears = len(speed_per_gear)

    gs_style = min(gs_style, 1)
    gs_style = max(gs_style, 0)

    gs = []

    for gear in range(n_gears - 1):
        speed_by_gs = speed_per_gear[gear][-1] * gs_style + \
                      speed_per_gear[gear][0] * (1 - gs_style)
        speed_for_continuity = speed_per_gear[gear + 1][0]
        cutoff_s = max(speed_by_gs, speed_for_continuity)

        gs.append(cutoff_s)

    return gs


dsp.add_function(
    function_id='define_idle_engine_speed',
    function=sh.bypass,
    inputs=['idle_engine_speed_median', 'idle_engine_speed_std'],
    outputs=['idle_engine_speed']
)


@sh.add_function(dsp, outputs=['Tans'])
def find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop):
    """
    Gets coefficients and speed boundaries and returns Tans value for per speed
    per gear.

    :param coefs_per_gear:
        Coefficients per gear.
    :type coefs_per_gear: list

    :param Start:
        Start speed per gear.
    :type Start: list

    :param Stop:
        Stop speed per gear.
    :type Stop: list

    :return:
        Tan.
    :rtype: list
    """
    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    Tans = []

    for gear, coefs in enumerate(coefs_per_gear):
        x_new = np.arange(Start[gear], Stop[gear], 0.1)
        a_new = np.array([np.dot(coefs, np.power(i, vars)) for i in x_new])

        Tans.append(np.diff(a_new) * 10)

    return Tans


def _find_gs_cut_tans(tmp_min, tmp_max, tan, tmp_min_next, gs_style):
    """

    Find where gear is changed, vased on tans and gs_style

    :param tmp_min:
    :param tmp_max:
    :param tan:
    :param tmp_min_next:
    :param cutoff:
    :return:
    """
    max_tan = np.max(tan)
    min_tan = np.min(tan)
    acc_range = max_tan - min_tan

    # tan starts from positive and goes negative, so I use (1 - cutoff)
    # for the percentage
    if gs_style > 0.99:
        gs_style = 1
    elif gs_style < 0.01:
        gs_style = 0.01
    tan_cutoff = (1 - gs_style) * acc_range + min_tan

    # Search_from = int(tmp_min_next * 10)
    search_from = int((tmp_min_next - tmp_min) * 10) + 1

    i_cut = len(tan) - 1
    while tan[i_cut] < tan_cutoff and i_cut >= search_from:
        i_cut -= 1

    gear_cut = tmp_min + i_cut / 10 + 0.1

    return gear_cut


@sh.add_function(dsp, inputs_kwargs=True, outputs=['gs'])
def gear_points_from_tan(Tans, gs_style, Start, Stop, use_linear_gs=False):
    """
    Get the gear cuts based on gear shifting style and tangent values.

    :param Tans:
        Tangent values per gear.
    :type Tans:

    :param gs_style:
        Gear shifting style.
    :type gs_style:

    :param Start:
        Start speed per gear curve.
    :type Start:

    :param Stop:
        Stop speed per gear curve.
    :type Stop:

    :param use_linear_gs:
        Use gear linear to calculate gs.
    :type use_linear_gs: bool

    :return:
        Gear limits
    :type:
    """
    if use_linear_gs:
        return sh.NONE
    n_gears = len(Tans)
    gs_cut = [gs_style for i in range(n_gears)]

    gs = []

    for i in range(n_gears - 1):
        tmp_min = Start[i]
        tmp_max = Stop[i]
        tan = Tans[i]
        tmp_min_next = Start[i + 1]
        cutoff_s = _find_gs_cut_tans(tmp_min, tmp_max, tan, tmp_min_next,
                                     gs_cut[i])

        gs.append(cutoff_s)

    return gs


@sh.add_function(dsp, outputs=['times'])
def define_times(duration, sim_step):
    return np.arange(0, duration + sim_step, sim_step)


@sh.add_function(dsp, outputs=['gears', 'velocities'])
def run_simulation(transmission, v_start, gs, times, Curves, v_des,
                   driver_style):
    """
    Run simulation.

    :param transmission:
        Transmission type of vehicle.
    :type transmission: str

    :param v_start:
        Current speed.
    :type v_start: int

    :param sim_step:
        Simulation step in seconds.
    :type sim_step: float

    :param gs: list
        Gear shifting style.
    :type gs: int

    :param times:
        Sample time series.
    :type times: np.array

    :param Curves: list
        Final acceleration curves.
    :type Curves: list

    :param v_des:
        Desired velocity.
    :type v_des: int

    :param driver_style:
        Driving style.
    :type driver_style: int

    :return:
        Speeds & Acceleration
    :rtype: list, list
    """
    from .simulation import gear_for_speed_profiles, simulation_step_function
    velocities = [v_start]

    speed = v_start

    # Returns the gear that must be used and the clutch condition
    gear, gear_count = gear_for_speed_profiles(gs, speed, 0, 0)
    gear_count = 0
    gears = [gear]

    # Core loop
    for dt in np.diff(times):
        speed, gear, gear_count = simulation_step_function(
            transmission, speed, gear, gear_count, gs, Curves, v_des,
            driver_style, dt
        )

        # Gather data
        gears.append(gear)
        velocities.append(speed)
    return gears, velocities


@sh.add_function(dsp, outputs=['accelerations'])
def calculate_accelerations(times, velocities):
    dv = np.ediff1d(velocities, to_begin=[0])
    dt = np.ediff1d(times, to_begin=[1])
    return (dv / dt).tolist()


if __name__ == '__main__':
    dsp.plot()