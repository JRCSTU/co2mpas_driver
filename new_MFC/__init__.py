import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
import copy
import numpy as np
import schedula as sh
from scipy.interpolate import CubicSpline, interp1d
from new_MFC.co2mpas import get_full_load, \
    calculate_full_load_speeds_and_powers, estimate_f_coefficients

from new_MFC.gear_functions import create_clutch_list, gear_for_speed_profiles
from new_MFC.vehicle_specs_class import HardcodedParams

from new_MFC.generic_co2mpas import light_co2mpas_instant
from new_MFC.functions import calculate_wheel_power, calculate_wheel_speeds, \
    calculate_wheel_torques, calculate_final_drive_speeds_in, \
    calculate_final_drive_torque_losses_v1, calculate_final_drive_torques_in, \
    calculate_gear_box_speeds_in_v1, create_gearbox_params, gear_box_torques_in, \
    calculate_brake_mean_effective_pressures, mean_piston_speed, parameters, \
    calculate_fuel_ABC, calculate_VMEP, calc_fuel_consumption, \
    calculate_gear_box_power_out

dsp = sh.Dispatcher()
dsp.add_func(get_full_load, outputs=['full_load_curve'])
dsp.add_func(
    calculate_full_load_speeds_and_powers,
    outputs=['full_load_speeds', 'full_load_powers']
)
dsp.add_func(
    estimate_f_coefficients,
    outputs=['road_loads']
)
dsp.add_func(
    calculate_wheel_power,
    outputs=['veh_wheel_power']
)
dsp.add_func(
    calculate_wheel_speeds,
    outputs=['veh_wheel_speed']
)
dsp.add_func(
    calculate_wheel_torques,
    outputs=['veh_wheel_torque']
)
dsp.add_func(
    calculate_final_drive_speeds_in,
    outputs=['final_drive_speed']
)
dsp.add_func(
    calculate_final_drive_torque_losses_v1,
    outputs=['final_drive_torque_losses']
)
dsp.add_func(
    calculate_final_drive_torques_in,
    outputs=['final_drive_torque_in']
)
dsp.add_func(
    calculate_gear_box_speeds_in_v1,
    outputs=['gear_box_speeds_in']
)
dsp.add_func(
    create_gearbox_params,
    outputs=['gearbox_params']
)
dsp.add_func(
    gear_box_torques_in,
    outputs=['gear_box_torques_in_']
)
dsp.add_func(
    calculate_gear_box_power_out,
    outputs=['gear_box_power_out']
)
dsp.add_func(
    calculate_brake_mean_effective_pressures,
    outputs=['br_eff_pres'],
    inputs=['gear_box_speeds_in, gear_box_power_out', 'fuel_eng_capacity', 'min_engine_on_speed']
)
dsp.add_func(
    mean_piston_speed,
    outputs=['engine_cm']
)
dsp.add_func(
    parameters,
    outputs=['params']
)
dsp.add_func(
    calculate_fuel_ABC,
    outputs=['fuel_A', 'fuel_B', 'fuel_C']
)
dsp.add_func(
    calculate_VMEP,
    outputs=['VMEP']
)
dsp.add_func(
    calc_fuel_consumption,
    outputs=['fc']
)


@sh.add_function(dsp, outputs=['full_load_torque'])
def calculate_full_load_torque(full_load_powers, full_load_speeds):
    """
    Full load curves of speed and torque.

    :param full_load_powers:
        Engine ignition type (positive or compression).
    :type full_load_powers: str

    :param full_load_speeds:
        Engine nominal power [kW].
    :type full_load_speeds: float
    :return: full_load_torque
    """
    full_load_torque = full_load_powers * 1000 * (full_load_speeds / 60 * 2 * np.pi) ** -1

    return full_load_torque


# Speed and acceleration ranges and points for each gear
@sh.add_function(dsp, outputs=['speed_per_gear', 'acc_per_gear'])
def get_speeds_n_accelerations_per_gear(gr, idle_engine_speed, tire_radius,
                                        driveline_slippage, final_drive,
                                        driveline_efficiency, veh_mass,
                                        full_load_speeds, full_load_torque):
    """
    Speed and acceleration points per gear are calculated based on
    full load curve, new version works with array and
    forbid acceleration over the maximum vehicle speed

    :param gr:
    :type gr: list
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
    :param full_load_torque:
    :type full_load_torque: ndarray
    :return: speed_per_gear
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
def get_start_stop(gr, veh_max_speed, speed_per_gear, acc_per_gear,
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
    car_res_curve, car_res_curve_force = veh_resistances(f0, f1, f2, list(sp_bins),
                                                         veh_mass)
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


@sh.add_function(dsp, outputs=['cs_acc_per_gear'])
def get_cubic_splines_of_speed_acceleration_relationship(gr, speed_per_gear,
                                                         acc_per_gear):
    """
    Based on speed/acceleration points per gear, cubic splines are calculated
    (old MFC)

    :param gr:
    :param speed_per_gear:
    :param acc_per_gear:
    :return:
    """
    cs_acc_per_gear = []
    for j in range(len(gr)):
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


@sh.add_function(dsp, outputs=['fp'])
def light_co2mpas_series(gearbox_type, veh_params, gb_type, car_type, veh_mass,
                         r_dynamic, final_drive, gr, engine_max_torque,
                         max_power, fuel_eng_capacity, fuel_engine_stroke,
                         fuel_type, fuel_turbo, road_loads, sp, gs, sim_step, **kwargs):
    """
    :param gearbox_type:
    :param veh_params:
    :param gb_type:
    :param car_type:
    :param veh_mass:
    :param r_dynamic:
    :param final_drive:
    :param gr:
    :param engine_max_torque:
    :param max_power:
    :param fuel_eng_capacity:
    :param fuel_engine_stroke:
    :param fuel_type:
    :param fuel_turbo:
    :param road_loads:
    :param sp:          In km/h!!!
    :param gs:
    :param sim_step:    in sec
    :return:
    """

    gear_list = {}
    clutch_list = []
    gear_list_flag = False
    if 'gear_list' in kwargs:
        gear_list_flag = True
        gear_list = kwargs['gear_list']
        if 'clutch_duration' in kwargs:
            clutch_duration = kwargs['clutch_duration']
        else:
            clutch_duration = int(0.5 % sim_step)
        clutch_list = create_clutch_list(gear_list, clutch_duration)

    hardcoded_params = HardcodedParams()

    # n_wheel_drive = my_car.car_type
    # road_loads = estimate_f_coefficients(veh_mass, type_of_car, car_width,
    #                                      car_height, passengers=0)

    slope = 0
    # FIX First convert km/h to m/s in order to have acceleration in m/s^2
    ap = np.diff([i / (3.6 * sim_step) for i in sp])

    # gear number and gear count for shifting duration
    # simulated_gear = [0, 30]
    fp = []

    if gearbox_type == 'manual':
        veh_params = hardcoded_params.params_gearbox_losses['Manual']
        gb_type = 0
    else:
        veh_params = hardcoded_params.params_gearbox_losses['Automatic']
        gb_type = 1

    # gear is the current gear and gear_count counts the time-steps
    # in order to prevent continuous gear shifting.
    gear = 0
    # Initializing gear count.
    gear_count = 30

    for i in range(1, len(sp)):
        speed = sp[i]
        acceleration = ap[i - 1]

        if gear_list_flag:
            gear = gear_list[i]
            gear_count = clutch_list[i]
        else:
            gear, gear_count = gear_for_speed_profiles(gs, speed / 3.6, gear,
                                                       gear_count, gb_type)
        fc = light_co2mpas_instant(veh_mass, r_dynamic, car_type, final_drive,
                                   gr, veh_params, engine_max_torque,
                                   fuel_eng_capacity, speed, acceleration,
                                   max_power, fuel_engine_stroke, fuel_type,
                                   fuel_turbo, hardcoded_params, road_loads,
                                   slope, gear, gear_count, sim_step)

        fp.append(fc)

    return fp


if __name__ == '__main__':
    dsp.plot()
