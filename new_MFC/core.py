import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
import copy
import numpy as np
import schedula as sh
from new_MFC.co2mpas import get_full_load, \
    calculate_full_load_speeds_and_powers, estimate_f_coefficients, \
    calculate_full_load_torques

dsp = sh.Dispatcher()
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
        gear_box_ratios, idle_engine_speed, tire_radius, driveline_slippage,
        final_drive, driveline_efficiency, veh_mass, full_load_speeds,
        full_load_torques):
    """
    Speed and acceleration points per gear are calculated based on
    full load curve, new version works with array and
    forbid acceleration over the maximum vehicle speed

    :param gear_box_ratios:
    :type gear_box_ratios: list
    :param idle_engine_speed:
    :type idle_engine_speed: tuple
    :param tire_radius:
    :type tire_radius: float
    :param driveline_slippage:
    :type driveline_slippage: int
    :param final_drive:
    :type final_drive: float
    :param driveline_efficiency:
    :type driveline_efficiency: float
    :param veh_mass:
    :type veh_mass: float
    :param full_load_speeds:
    :type full_load_speeds: ndarray
    :param full_load_torques:
    :type full_load_torques: ndarray
    :return: speed_per_gear
    """
    speed_per_gear, acc_per_gear = [], []

    full_load_speeds = np.array(full_load_speeds)
    full_load_torques = np.array(full_load_torques)

    for j in range(len(gear_box_ratios)):
        mask = full_load_speeds > 1.25 * idle_engine_speed[0]

        temp_speed = 2 * np.pi * tire_radius * full_load_speeds[mask] * (
                1 - driveline_slippage) / (
                             60 * final_drive * gear_box_ratios[j])
        speed_per_gear.append(temp_speed)

        temp_acc = full_load_torques[mask] * (final_drive * gear_box_ratios[
            j]) * driveline_efficiency / (
                           tire_radius * veh_mass)

        acc_per_gear.append(temp_acc)

    return speed_per_gear, acc_per_gear


@sh.add_function(dsp, outputs=['coefs_per_gear'])
def get_tan_coefs(speed_per_gear, acc_per_gear, degree):
    """
    Full load curve is fitted to a polynomial of degree

    :param speed_per_gear:
    :param acc_per_gear:
    :param degree:
    :return: coefs_per_gear: the coefficients of the polynomial for each gear
    """

    coefs_per_gear = []
    for speeds, acceleration in zip(speed_per_gear, acc_per_gear):
        fit_coef = np.polyfit(speeds, acceleration, degree)
        coefs_per_gear.append(fit_coef)

    return coefs_per_gear


@sh.add_function(dsp, outputs=['cs_acc_per_gear', 'Start', 'Stop'])
def ev_curve(engine_max_power, tire_radius, driveline_slippage,
             motor_max_torque, final_drive, gear_box_ratios,
             driveline_efficiency, veh_mass, veh_max_speed):
    """

    Full load curve of EVs (based on Yinglong)

    :param engine_max_power:
    :param tire_radius:
    :param driveline_slippage:
    :param motor_max_torque:
    :param final_drive:
    :param gear_box_ratios:
    :param driveline_efficiency:
    :param veh_mass:
    :param veh_max_speed:
    :return:
    """

    from scipy.interpolate import CubicSpline

    motor_base_speed = engine_max_power * 1000 * (motor_max_torque
                                                  / 60 * 2 * np.pi) ** -1  # rpm
    # motor_max_speed = my_car.veh_max_speed * (60 * my_car.final_drive * my_car.gr) / (1 - my_car.driveline_slippage) / (2 * np.pi * my_car.tire_radius)  # rpm
    veh_base_speed = 2 * np.pi * tire_radius * motor_base_speed * \
                     (1 - driveline_slippage) / (
                             60 * final_drive * gear_box_ratios)  # m/s
    veh_max_acc = motor_max_torque * (
            final_drive * gear_box_ratios) * driveline_efficiency / (
                          tire_radius * veh_mass)  # m/s2

    speeds = np.arange(0, veh_max_speed + 0.1, 0.1)  # m/s

    with np.errstate(divide='ignore'):
        accelerations = engine_max_power * 1000 * \
                        driveline_efficiency / (speeds * veh_mass)

    accelerations[accelerations > veh_max_acc] = veh_max_acc
    accelerations[accelerations < 0] = 0

    cs_acc_ev = CubicSpline(speeds, accelerations)
    start = 0
    stop = veh_max_speed

    return [cs_acc_ev], [start], [stop]


@sh.add_function(dsp, inputs_kwargs=True, outputs=['poly_spline'])
def get_cubic_splines_of_speed_acceleration_relationship(
        speed_per_gear, acc_per_gear, use_cubic=True):
    """
    Based on speed/acceleration points per gear, cubic splines are calculated
    (old MFC)

    :param gr:
    :param speed_per_gear:
    :param acc_per_gear:
    :return:
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
    Use the coefs to get a "spline" that could be used.
    AT TIME IT IS USED AS calculate_curve_to_use FUNCTION IS USING SPLINES

    :param coefs_per_gear:
    :param starting_speed:
    :return:
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


# Start/stop speed for each gear
@sh.add_function(dsp, outputs=['Start', 'Stop'])
def get_start_stop(gear_box_ratios, veh_max_speed, speed_per_gear, acc_per_gear,
                   poly_spline):
    """
    Calculate Speed boundaries for each gear

    :param gr:
    :param veh_max_speed:
    :param speed_per_gear:
    :param acc_per_gear:
    :param poly_spline:
    :return:
    """
    speed_per_gear = copy.deepcopy(speed_per_gear)
    acc_per_gear = copy.deepcopy(acc_per_gear)
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
                    poly_spline[j + 1](speed_per_gear[j][k]) > poly_spline[j](
                speed_per_gear[j][k])):
                max_point = k
                speed_per_gear[j] = speed_per_gear[j][:max_point]
                acc_per_gear[j] = acc_per_gear[j][:max_point]
                break

    # The limits of the gears that should be provided to the gear shifting model
    Start = []
    Stop = []
    for i in speed_per_gear:
        Start.append(i[0])
        Stop.append(min(i[-1], veh_max_speed))
    Start[0] = 0
    return Start, Stop


# Start/stop speed for each gear
@sh.add_function(dsp, outputs=['sp_bins'])
def define_sp_bins(Stop):
    return np.arange(0, Stop[-1] + 1, 0.01)


# Calculate Curves
@sh.add_function(dsp, outputs=['car_res_curve', 'car_res_curve_force',
                               'Alimit'])
def get_resistances(type_of_car, car_type, veh_mass, engine_max_power,
                    car_width, car_height, sp_bins):
    """
    Calculate resistances and return spline

    :param type_of_car:
    :param car_type:
    :param veh_mass:
    :param engine_max_power:
    :param car_type:
    :param car_width:
    :param car_height:
    :param sp_bins:
    :return:
    """

    from .co2mpas import estimate_f_coefficients, veh_resistances, Armax
    f0, f1, f2 = estimate_f_coefficients(veh_mass, type_of_car, car_width,
                                         car_height)
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2,
                                                         list(sp_bins),
                                                         veh_mass)
    Alimit = Armax(car_type, veh_mass, engine_max_power)
    return car_res_curve, car_res_curve_force, Alimit


@sh.add_function(dsp, outputs=['Curves'])
def calculate_curves_to_use(poly_spline, Start, Stop, Alimit, car_res_curve,
                            sp_bins):
    """
    Get the final speed acceleration curves based on full load curves and
    resistances for all curves

    :param poly_spline:
    :param acc:
    :param Alimit:
    :param car_res_curve:
    :param Start:
    :param Stop:
    :param sp_bins:
    :return:
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
    strategy

    :param speed_per_gear:
    :param gs_style:
    :param use_linear_gs:
    :type use_linear_gs: bool
    :return:
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


@sh.add_function(dsp, outputs=['Tans'])
def find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop):
    """
    Gets coefficients and speed boundaries and returns Tans value for per speed
    per gear

    :param coefs_per_gear:
    :param Start:
    :param Stop:
    :return:
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

    :param Tans: tangent values per gear.
    :param gs_style: Gear shifting style
    :param Start: Start speed per gear curve.
    :param Stop: Stop speed per gear curve.
    :return:
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


if __name__ == '__main__':
    dsp.plot()
