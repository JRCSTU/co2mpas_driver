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


def calculate_full_load_speeds_and_powers(full_load_curve, my_car):
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
    full_load_powers = full_load_curve(n_norm) * my_car.engine_max_power
    idle = my_car.idle_engine_speed[0]
    full_load_speeds = n_norm * (my_car.engine_max_speed_at_max_power - idle) + idle

    return full_load_speeds, full_load_powers


# The maximum force that the vehicle can have on the road
def Armax(my_car, road_type=1):
    """
    Calculating the maximum acceleration possible for the vehicle object my_car,
    under road_type conditions.

    :param my_car: vehicle specs object
    :param road_type: road condition (1: normal, 2: wet, 3: icy)
    :return:
    """
    if my_car.car_type == 2:  # forward-wheel drive vehicles
        fmass = 0.6 * my_car.veh_mass
    elif my_car.car_type == 4:  # rear-wheel drive vehicles
        fmass = 0.45 * my_car.veh_mass
    else:  # all-wheel drive vehicles, 4x4
        fmass = 1 * my_car.veh_mass

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
    mh = mh_base * (alpha * np.log(my_car.engine_max_power) + beta) / 190

    # * cos(f) for the gradient of the road. Here we consider as 0
    Frmax = fmass * 9.8066 * mh

    return Frmax / my_car.veh_mass


# Calculates a spline with the resistances
def veh_resistances(f0, f1, f2, sp, total_mass):
    """
    Return the resistances that a vehicle faces, per speed

    :param f0:
    :type f0: float
    :param f1:
    :type f1: float
    :param f2:
    :type f2:float
    :param sp:
    :type sp: list
    :param total_mass:
    :type total_mass: float
    :return: resistance_spline_curve, resistance_spline_curve_f
    :rtype: CubicSpline, CubicSpline
    """
    Fresistance = []
    for i in range(len(sp)):
        Fresistance.append(f0 + f1 * sp[i] * 3.6 + f2 * pow(sp[i] * 3.6, 2))
        # Facc = Fmax @ wheel - f0 * cos(a) - f1 * v - f2 * v2 - m * g * sin(a)

    aprx_mass = int(total_mass)
    Aresistance = [x / aprx_mass for x in Fresistance]
    a = int(np.floor(sp[0]))
    b = int(np.floor(sp[-1]))
    resistance_spline_curve = CubicSpline([k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)], \
                                          [Aresistance[0]] * 10 + Aresistance + [Aresistance[-1]] * 10)
    resistance_spline_curve_f = CubicSpline([k for k in range(a - 10, a)] + sp + [k for k in range(b + 1, b + 11)],
                                            [Fresistance[0]] * 10 + Fresistance + [Fresistance[-1]] * 10)

    return resistance_spline_curve, resistance_spline_curve_f


def estimate_f_coefficients(my_car, passengers=0):
    """
    f0, f1, f2 coefficients of resistance are estimated

    :param my_car:
    :param passengers:
    :type passengers: int
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
    theor_aero_coeff = d[my_car.type_of_car]

    operating_mass = my_car.veh_mass + 100 + 75 * passengers
    f0 = (operating_mass + 100) * rolling_res_coef * 9.81
    f2 = 0.5 * 1.2 * (0.84 * my_car.car_width * my_car.car_height * theor_aero_coeff) / pow(3.6, 2)
    f1 = -71.735 * f2 + 2.7609

    return f0, f1, f2


def get_load_speed_n_torque(my_car):
    """
    :param my_car:
    :return:
        full_load_speeds, full_load_torques
    :rtype: numpy.array, numpy.array
    """

    full_load = get_full_load(my_car.ignition_type)
    full_load_speeds, full_load_powers = calculate_full_load_speeds_and_powers(full_load, my_car)
    full_load_torques = full_load_powers * 1000 * (full_load_speeds / 60 * 2 * np.pi) ** -1
    return full_load_speeds, full_load_torques


def get_speeds_n_accelerations_per_gear(my_car, full_load_speeds,
                                        full_load_torques):
    """
    Speed and acceleration points per gear are calculated based on
    full load curve, new version works with array and
    forbid acceleration over the maximum vehicle speed

    :param my_car:
    :param full_load_speeds:
    :type full_load_speeds: numpy.array
    :param full_load_torques:
    :type full_load_torques: numpy.array
    :return:
        speed_per_gear, acc_per_gear
    :rtype: list, list
    """
    speed_per_gear, acc_per_gear = [], []

    full_load_speeds = np.array(full_load_speeds)
    full_load_torques = np.array(full_load_torques)

    for j in range(len(my_car.gr)):
        mask = full_load_speeds > 1.25 * my_car.idle_engine_speed[0]

        temp_speed = 2 * np.pi * my_car.tire_radius * full_load_speeds[mask] * (1 - my_car.driveline_slippage) / (
            60 * my_car.final_drive * my_car.gr[j])
        speed_per_gear.append(temp_speed)

        temp_acc = full_load_torques[mask] * (my_car.final_drive * my_car.gr[j]) * my_car.driveline_efficiency / (
            my_car.tire_radius * my_car.veh_mass)

        acc_per_gear.append(temp_acc)

    return speed_per_gear, acc_per_gear


def get_cubic_splines_of_speed_acceleration_relationship(my_car, speed_per_gear, acc_per_gear):
    """
    Based on speed/acceleration points per gear, cubic splines are calculated
    (old MFC)

    :param my_car:
    :param speed_per_gear:
    :param acc_per_gear:
    :return:
    """
    cs_acc_per_gear = []
    for j in range(len(my_car.gr)):
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


def get_start_stop(my_car, speed_per_gear, acc_per_gear, poly_spline):
    """

    Calculate Speed boundaries for each gear

    :param my_car:
    :param speed_per_gear:
    :type speed_per_gear: list
    :param acc_per_gear:
    :type acc_per_gear: list
    :param poly_spline:
    :type poly_spline: list
    :return: Start, Stop
    :rtype: list, list
    """
    import copy
    speed_per_gear = copy.deepcopy(speed_per_gear)
    acc_per_gear = copy.deepcopy(acc_per_gear)
    # To ensure that a higher gear starts from higher speed
    for j in range(len(my_car.gr) - 1, 0, -1):
        for k in range(len(speed_per_gear[j])):
            if speed_per_gear[j - 1][0] < speed_per_gear[j][0]:
                break
            else:
                # If the gear ratios are not declining,
                # there is an error in the database. Return error.
                return
                # speed_per_gear[j] = speed_per_gear[j][3:]

    # Find where the curve of each gear cuts the next one.
    for j in range(len(my_car.gr) - 1):
        for k in range(np.minimum(len(speed_per_gear[j]), len(speed_per_gear[j + 1]))):
            if (speed_per_gear[j][k] > speed_per_gear[j + 1][0]) & (
                        poly_spline[j + 1](speed_per_gear[j][k]) > poly_spline[j](speed_per_gear[j][k])):
                max_point = k
                speed_per_gear[j] = speed_per_gear[j][:max_point]
                acc_per_gear[j] = acc_per_gear[j][:max_point]
                break

    # The limits of the gears that should be provided to the gear shifting model
    Start = []
    Stop = []
    for i in speed_per_gear:
        Start.append(i[0])
        Stop.append(min(i[-1], my_car.veh_max_speed))
    Start[0] = 0
    return Start, Stop


def get_resistances(my_car, sp_bins):
    """
    Calculate resistances and return spline.

    :param my_car:
    :param sp_bins:
    :type sp_bins: np.array
    :return: car_res_curve, car_res_curve_force, Alimit
    :rtype: CubicSpline, CubicSpline, float
    """
    f0, f1, f2 = estimate_f_coefficients(my_car, 0)
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2, list(sp_bins), my_car.veh_mass)
    Alimit = Armax(my_car)
    return car_res_curve, car_res_curve_force, Alimit


def calculate_curves_to_use(poly_spline, Start, Stop, Alimit, car_res_curve,
                            sp_bins):
    """

    Get the final speed acceleration curves based on full load curves and resistances for all curves

    :param poly_spline:
    :type poly_spline: list
    :param Start:
    :type Start: list
    :param Stop:
    :type Stop: limit
    :param Alimit:
    :type Alimit: float
    :param car_res_curve:
    :type car_res_curve: CubicSpline
    :param sp_bins:
    :type sp_bins: numpy.array
    :return: Curves
    :rtype: list
    """
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


def calculate_dec_curves_to_use(sp_bins):
    '''
    Calculates deceleration curves.
    :param sp_bins:
    :return:
    '''
    ppar = [0.0045, -0.1710, -1.8835]
    dec_curves = np.poly1d(ppar)
    final_dec = []
    Curves_dec = []
    for k in range(len(sp_bins)):
        final_dec.append(dec_curves(sp_bins[k]))
    from scipy.interpolate import interp1d
    Curves_dec.append(interp1d(sp_bins, final_dec))

    return Curves_dec


def get_starting_speed(speed_per_gear):
    starting_speed = speed_per_gear[0][0]
    return starting_speed


def get_tan_coefs(speed_per_gear, acc_per_gear, degree):
    """
    Full load curve is fitted to a polynomial of degree

    :param speed_per_gear:
    :type speed_per_gear: list
    :param acc_per_gear:
    :type acc_per_gear: list
    :param degree:
    :return: coefs_per_gear:
        the coefficients of the polynomial for each gear
    :rtype: list
    """

    coefs_per_gear = []
    for speeds, acceleration in zip(speed_per_gear, acc_per_gear):
        fit_coef = np.polyfit(speeds, acceleration, degree)
        coefs_per_gear.append(fit_coef)

    return coefs_per_gear


def get_spline_out_of_coefs(coefs_per_gear, starting_speed):
    """
    Use the coefs to get a "spline" that could be used.
    AT TIME IT IS USED AS calculate_curve_to_use FUNCTION IS USING SPLINES

    :param coefs_per_gear:
    :type coefs_per_gear: list
    :param starting_speed:
    :type starting_speed: float
    :return:
        spline_from_poly
    :rtype: list
    """

    degree = len(coefs_per_gear[0]) - 1
    vars = np.arange(degree, -1, -1)

    spline_from_poly = []

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
    spline_from_poly.append(interp1d(x_new, a_new, fill_value='extrapolate'))

    for fit_coef in coefs_per_gear[1:]:
        x_new = np.arange(0, 70, 0.1)
        a_new = np.array([np.dot(fit_coef, np.power(i, vars)) for i in x_new])
        spline_from_poly.append(interp1d(x_new, a_new, fill_value='extrapolate'))

    return spline_from_poly


def ev_curve(my_car):
    """
    Full load curve of EVs (based on Yinglong)

    :param my_car:
    :return:
    """

    veh_max_acc = my_car.motor_max_torque * (
            my_car.final_drive * my_car.gr) * my_car.driveline_efficiency / (
                my_car.tire_radius * my_car.veh_mass)  # m/s2

    speeds = np.arange(0, my_car.veh_max_speed + 0.1, 0.1)  # m/s

    with np.errstate(divide='ignore'):
        accelerations = my_car.engine_max_power * 1000 * my_car.driveline_efficiency / (speeds * my_car.veh_mass)

    accelerations[accelerations > veh_max_acc] = veh_max_acc
    accelerations[accelerations < 0] = 0

    cs_acc_ev = CubicSpline(speeds, accelerations)
    start = 0
    stop = my_car.veh_max_speed

    return [cs_acc_ev], [start], [stop]
