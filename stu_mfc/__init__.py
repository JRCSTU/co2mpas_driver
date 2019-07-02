import schedula as sh
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

dsp = sh.Dispatcher()


@sh.add_function(dsp, outputs=['full_load_speeds', 'full_load_torque'])
def get_load_speed_n_torque(
        ignition_type, engine_max_power, engine_max_speed_at_max_power,
        idle_engine_speed):
    """
    Full load curves of speed and torque.

    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str

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
    """
    from .co2mpas import get_full_load, calculate_full_load_speeds_and_powers
    full_load = get_full_load(ignition_type)
    full_load_speeds, full_load_powers = calculate_full_load_speeds_and_powers(
        full_load, engine_max_power, engine_max_speed_at_max_power,
        idle_engine_speed
    )
    full_load_torque = full_load_powers * 1000 * (full_load_speeds / 60 * 2 * np.pi) ** -1
    return full_load_speeds, full_load_torque


# Speed and acceleration ranges and points for each gear
@sh.add_function(dsp, outputs=['speed_per_gear', 'acc_per_gear'])
def get_speeds_n_accelerations_per_gear(gr, idle_engine_speed, tire_radius,
            driveline_slippage, final_drive, driveline_efficiency, veh_mass,
            full_load_speeds, full_load_torque):
    """
    Speed and acceleration points per gear are calculated based on
    full load curve, new version works with array and
    forbid acceleration over the maximum vehicle speed

    :param gr:
    :param idle_engine_speed:
    :param tire_radius:
    :param driveline_slippage:
    :param final_drive:
    :param driveline_efficiency:
    :param veh_mass:
    :param full_load_speeds:
    :param full_load_torque:
    :return:
    """
    speed_per_gear, acc_per_gear = [], []

    full_load_speeds = np.array(full_load_speeds)
    full_load_torque = np.array(full_load_torque)

    for j in range(len(gr)):
        mask = full_load_speeds > 1.25 * idle_engine_speed[0]

        temp_speed = 2 * np.pi * tire_radius * full_load_speeds[mask] * (
                    1 - driveline_slippage) / (
                             60 * final_drive * gr[j])
        speed_per_gear.append(temp_speed)

        temp_acc = full_load_torque[mask] * (final_drive * gr[
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


@sh.add_function(dsp, outputs=['poly_spline'])
def get_spline_out_of_coefs(coefs_per_gear, starting_speed):
    """
    Use the coefs to get a "spline" that could be used.
    AT TIME IT IS USED AS calculate_curve_to_use FUNCTION IS USING SPLINES

    :param coefs_per_gear:
    :param starting_speed:
    :return:
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


# Start/stop speed for each gear
@sh.add_function(dsp, outputs=['Start', 'Stop'])
def get_start_stop(gr, veh_max_speed, speed_per_gear, acc_per_gear,
                   cs_acc_per_gear):
    """
    Calculate Speed boundaries for each gear

    :param gr:
    :param veh_max_speed:
    :param speed_per_gear:
    :param acc_per_gear:
    :return:
    """
    # To ensure that a higher gear starts from higher speed
    for j in range(len(gr) - 1, 0, -1):
        for k in range(len(speed_per_gear[j])):
            if speed_per_gear[j - 1][0] < speed_per_gear[j][0]:
                break
            else:
                # If the gear ratios are not declining,
                # there is an error in the database. Return error.
                return
                # speed_per_gear[j] = speed_per_gear[j][3:]

    # Find where the curve of each gear cuts the next one.
    for j in range(len(gr) - 1):
        for k in range(np.minimum(len(speed_per_gear[j]), len(speed_per_gear[j + 1]))):
            if (speed_per_gear[j][k] > speed_per_gear[j + 1][0]) & (
                        cs_acc_per_gear[j + 1](speed_per_gear[j][k]) > cs_acc_per_gear[j](speed_per_gear[j][k])):
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
@sh.add_function(dsp, outputs=['car_res_curve', 'car_res_curve_force', 'Alimit'])
def get_resistances(car_type, veh_mass, engine_max_power, car_width, car_height, sp_bins):
    """
    Calculate resistances and return spline

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
    f0, f1, f2 = estimate_f_coefficients(veh_mass, car_type, car_width, car_height)
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2, list(sp_bins), veh_mass)
    Alimit = Armax(car_type, veh_mass, engine_max_power)
    return car_res_curve, car_res_curve_force, Alimit


# Extract speed acceleration Splines
@sh.add_function(dsp, outputs=['gs'])
def gear_linear(speed_per_gear, gs_style):
    """
    Return the gear limits based on gs_style, using linear gear swifting
    strategy

    :param speed_per_gear:
    :param gs_style:
    :return:
    """
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


@sh.add_function(dsp, outputs=['Curves'])
def calculate_curves_to_use(cs_acc_per_gear, Start, Stop, Alimit, car_res_curve, sp_bins):
    """
    Get the final speed acceleration curves based on full load curves and resistances for all curves

    :param cs_acc_per_gear:
    :param acc:
    :param Alimit:
    :param car_res_curve:
    :param Start:
    :param Stop:
    :param sp_bins:
    :return:
    """
    Res = []

    for gear, acc in enumerate(cs_acc_per_gear):
        start = Start[gear] * 0.9
        stop = Stop[gear] + 0.1

        final_acc = acc(sp_bins) - car_res_curve(sp_bins)
        final_acc[final_acc > Alimit] = Alimit

        final_acc[(sp_bins < start)] = 0
        final_acc[(sp_bins > stop)] = 0
        final_acc[final_acc < 0] = 0

        Res.append(interp1d(sp_bins, final_acc))

    return Res


if __name__ == '__main__':
    dsp.plot()
