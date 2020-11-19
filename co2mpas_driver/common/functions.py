import math
import numpy


def calculate_wheel_power(velocities, acc, road_loads, veh_mass, slope):
    """
    Calculates the wheel power [kW].

    :param velocities:
        Velocity [km/h].
    :type velocities: numpy.array | float

    :param acc:
        Acceleration [m/s2].
    :type acc: numpy.array | float

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param veh_mass:
        Vehicle mass [kg].
    :type veh_mass: float

    :param slope:
        Slope.
    :type slope: float

    :return:
        Power at wheels [kW].
    :rtype: numpy.array | float

    ;param slope:
        Slope in tangent
    """

    f0, f1, f2 = road_loads

    # quadratic_term = f0 * math.cos(math.atan(slope)) + (f1 + f2 * velocities) * velocities
    #
    # vel = velocities / 3600

    quadratic_term = f0 * math.cos(math.atan(slope)) + (
            f1 + f2 * velocities) * velocities + 1.03 * veh_mass * acc + veh_mass * math.sin(
        math.atan(slope)) * 9.81

    res = float(quadratic_term * velocities) / 3600

    return res


def calculate_min_wheel_torque(velocities, accelerations, road_loads, vehicle_mass, f, r_dynamic):
    """
    Calculates the wheel power [kW].

    :param velocities:
        Velocity [km/h].
    :type velocities: numpy.array | float

    :param accelerations:
        Acceleration [m/s2].
    :type accelerations: numpy.array | float

    :param road_loads:
        Cycle road loads [N, N/(km/h), N/(km/h)^2].
    :type road_loads: list, tuple

    :param vehicle_mass:
        Vehicle mass [kg].
    :type vehicle_mass: float

    :param f:
        Tyre rolling resistance.
    :type f: float

    :param r_dynamic
        Tyre radius.
    :type r_dynamic: float

    :return:
        Power at wheels [kW].
    :rtype: numpy.array | float
    """

    f0, f1, f2 = road_loads

    quadratic_term = f0 * math.cos(f) + (f1 + f2 * velocities) * velocities

    vel = velocities / 3600

    return (quadratic_term + 1.03 * vehicle_mass * accelerations + vehicle_mass * math.sin(f)) * r_dynamic


def calculate_wheel_speeds(velocities, r_dynamic):
    """
    Calculates rotating speed of the wheels [RPM].

    :param velocities:
        Vehicle velocity [km/h].
    :type velocities: numpy.array | float

    :param r_dynamic:
        Dynamic radius of the wheels [m].
    :type r_dynamic: float

    :return:
        Rotating speed of the wheel [RPM].
    :rtype: numpy.array | float
    """

    return velocities * (30.0 / (3.6 * math.pi * r_dynamic))


def calculate_wheel_torques(wheel_powers, wheel_speeds):
    """
    Calculates torque at the wheels [N*m].

    :param wheel_powers:
        Power at the wheels [kW].
    :type wheel_powers: numpy.array | float

    :param wheel_speeds:
        Rotating speed of the wheel [RPM].
    :type wheel_speeds: numpy.array | float

    :return:
        Torque at the wheels [N*m].
    :rtype: numpy.array | float
    """

    pi = math.pi
    if isinstance(wheel_speeds, numpy.ndarray):
        return numpy.nan_to_num(wheel_powers / wheel_speeds * (30000.0 / pi))
    return wheel_powers / wheel_speeds * (30000.0 / pi) if wheel_speeds else 0.0


def calculate_final_drive_speeds_in(
        final_drive_speeds_out, final_drive_ratio_vector):
    """
    Calculates final drive speed [RPM].

    :param final_drive_speeds_out:
        Rotating speed of the wheel [RPM].
    :type final_drive_speeds_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :return:
        Final drive speed in [RPM].
    :rtype: numpy.array | float
    """

    return final_drive_speeds_out * final_drive_ratio_vector


def calculate_final_drive_torque_losses_v1(
        n_wheel_drive, final_drive_torques_out, final_drive_ratio_vector,
        final_drive_efficiency):
    """
    Calculates final drive torque losses [N*m].

    :param n_wheel_drive:
        Number of wheel drive [-].
    :type n_wheel_drive: int

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :param final_drive_efficiency:
        Final drive efficiency [-].
    :type final_drive_efficiency: float

    :return:
        Final drive torque losses [N*m].
    :rtype: numpy.array | float
    """

    eff_fd = final_drive_efficiency - (n_wheel_drive - 2) / 100
    to = final_drive_torques_out
    return (1 - eff_fd) / (eff_fd * final_drive_ratio_vector) * to


def calculate_final_drive_torques_in(
        final_drive_torques_out, final_drive_ratio_vector,
        final_drive_torque_losses):
    """
    Calculates final drive torque [N*m].

    :param final_drive_torques_out:
        Torque at the wheels [N*m].
    :type final_drive_torques_out: numpy.array | float

    :param final_drive_ratio_vector:
        Final drive ratio vector [-].
    :type final_drive_ratio_vector: numpy.array | float

    :param final_drive_torque_losses:
        Final drive torque losses [N*m].
    :type final_drive_torque_losses: numpy.array | float

    :return:
        Final drive torque in [N*m].
    :rtype: numpy.array | float
    """

    t = final_drive_torques_out / final_drive_ratio_vector

    return t + final_drive_torque_losses


def calculate_gear_box_speeds_in_v1(
        gear, final_drive_speed, gear_box_ratios, clutch):
    """
    Calculates Gear box speed vector [RPM].

    :param gears:
        Gear vector [-].
    :type gears: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_ratios:
        Gear box ratios [-].
    :type gear_box_ratios: dict

    :return:
        Gear box speed vector [RPM].
    :rtype: numpy.array
    """

    # d = {0: 0.0}
    #
    # d.update(gear_box_ratios)

    # ratios = numpy.vectorize(d.get)(gears)
    if clutch == 1:
        return 0

    return final_drive_speed * gear_box_ratios[gear - 1]


def gear_box_torques_in(
        min_engine_on_speed, gear_box_torques_out, gear_box_speeds_in,
        gear_box_speeds_out, par, clutch):
    """
    Calculates torque required according to the temperature profile [N*m].

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :param gear_box_torques_out:
        Torque gear_box vector [N*m].
    :type gear_box_torques_out: numpy.array

    :param gear_box_speeds_in:
        Engine speed vector [RPM].
    :type gear_box_speeds_in: numpy.array

    :param gear_box_speeds_out:
        Wheel speed vector [RPM].
    :type gear_box_speeds_out: numpy.array

    :param gear_box_efficiency_parameters_cold_hot:
        Parameters of gear box efficiency model:

            - `gbp00`,
            - `gbp10`,
            - `gbp01`
    :type gear_box_efficiency_parameters_cold_hot: dict

    :return:
        Torque required vector [N*m].
    :rtype: numpy.array
    """
    tgb, es, ws = gear_box_torques_out, gear_box_speeds_in, gear_box_speeds_out

    if gear_box_speeds_in != 0:
        ratio = gear_box_speeds_out / gear_box_speeds_in
    else:
        ratio = 0

    if tgb >= 0:
        if (es == 0 or ws == 0 or clutch == 1):
            res = 0
        else:
            res = (tgb * ratio - par['gbp10'] * es - par['gbp00']) / par['gbp01']
    else:
        res = -(-tgb * ratio * par['gbp01'] + par['gbp10'] * ws + par['gbp00'])

    return res


def create_gearbox_params(params, max_engine_torque):
    gbp10 = params['gbp10']['m'] * max_engine_torque + params['gbp10']['q']
    gbp00 = params['gbp00']['m'] * max_engine_torque + params['gbp00']['q']
    gbp01 = params['gbp01']['q']
    par = {'gbp00': gbp00, 'gbp10': gbp10, 'gbp01': gbp01}
    return par


def calculate_brake_mean_effective_pressures(
        engine_speeds_out, engine_powers_out, engine_capacity,
        min_engine_on_speed):
    """
    Calculates engine brake mean effective pressure [bar].

    :param engine_speeds_out:
        Engine speed vector [RPM].
    :type engine_speeds_out: numpy.array

    :param engine_powers_out:
        Engine power vector [kW].
    :type engine_powers_out: numpy.array

    :param engine_capacity:
        Engine capacity [cm3].
    :type engine_capacity: float

    :param min_engine_on_speed:
        Minimum engine speed to consider the engine to be on [RPM].
    :type min_engine_on_speed: float

    :return:
        Engine brake mean effective pressure vector [bar].
    :rtype: numpy.array
    """

    if engine_speeds_out > min_engine_on_speed:
        # res = float(engine_powers_out * 2000) / (engine_capacity * 0.1 * engine_speeds_out / 60)
        res = (engine_powers_out / engine_speeds_out) * 1200000.0 / engine_capacity
    else:
        res = 0
    #
    # p = numpy.zeros_like(engine_powers_out)
    # p[b] = engine_powers_out[b] / engine_speeds_out[b]
    # p[b] *= 1200000.0 / engine_capacity

    return res


def mean_piston_speed(engine_speed, fuel_engine_stroke):
    # stroke is in mm
    # engine_speed is in RPM

    res = engine_speed * fuel_engine_stroke / 30000.0
    return res


def parameters(max_power, capacity, eng_fuel, fuel_turbo):
    if eng_fuel == 'diesel':
        params = {
            'a': -0.0005 * max_power + 0.438451,
            'b': -0.26503 * (-0.0005 * max_power + 0.43845) + 0.12675,
            'c': -0.08528 * (-0.26503 * (-0.0005 * max_power + 0.43845) + 0.12675) + 0.0003,
            'a2': -0.0012,
            'b2': 0,
            'l': -1.55291,
            'l2': -0.0076,
            't': 3.5,  # 2.7,
            'trg': 85.
        }
    elif fuel_turbo == 'yes':
        params = {
            'a': 0.8882 * max_power / capacity + 0.377,
            'b': -0.17988 * (0.882 * max_power / capacity + 0.377) + 0.0899,
            'c': -0.06492 * (-0.17988 * (0.882 * max_power / capacity + 0.377) + 0.0899) + 0.000117,
            'a2': -0.00385,
            'b2': 0,
            'l': -2.14063,
            'l2': -0.00286,
            't': 3.5,  # 2.7,
            'trg': 85.
        }
    else:
        params = {
            'a': 0.8882 * max_power / capacity + 0.377,
            'b': -0.17988 * (0.882 * max_power / capacity + 0.377) + 0.0899,
            'c': -0.06492 * (-0.17988 * (0.882 * max_power / capacity + 0.377) + 0.0899) + 0.000117,
            'a2': -0.00266,
            'b2': 0,
            'l': -2.49882,
            'l2': -0.0025,
            't': 3.5,  # 2.7,
            'trg': 85.
        }

    return params


# def calculate_p0(params, engine_capacity, engine_stroke,
#                  idle_engine_speed_median, engine_fuel_lower_heating_value):
#     """
#     Calculates the engine power threshold limit [kW].
#
#     :param params:
#         CO2 emission model parameters (a2, b2, a, b, c, l, l2, t, trg).
#
#         The missing parameters are set equal to zero.
#     :type params: dict
#
#     :param engine_capacity:
#         Engine capacity [cm3].
#     :type engine_capacity: float
#
#     :param engine_stroke:
#         Engine stroke [mm].
#     :type engine_stroke: float
#
#     :param idle_engine_speed_median:
#         Engine speed idle median [RPM].
#     :type idle_engine_speed_median: float
#
#     :param engine_fuel_lower_heating_value:
#         Fuel lower heating value [kJ/kg].
#     :type engine_fuel_lower_heating_value: float
#
#     :return:
#         Engine power threshold limit [kW].
#     :rtype: float
#     """
#
#     engine_cm_idle = idle_engine_speed_median * engine_stroke / 30000.0
#
#     lhv = engine_fuel_lower_heating_value
#     FMEP = _calculate_fuel_mean_effective_pressure
#
#     engine_wfb_idle, engine_wfa_idle = FMEP(params, engine_cm_idle, 0, 1)
#     engine_wfa_idle = (3600000.0 / lhv) / engine_wfa_idle
#     engine_wfb_idle *= (3.0 * engine_capacity / lhv * idle_engine_speed_median)
#
#     return -engine_wfb_idle / engine_wfa_idle

def calculate_fuel_ABC(params, mean_piston_speed, n_powers, n_temperatures):
    p = params
    A = p['a2'] + p['b2'] * mean_piston_speed
    B = p['a'] + (p['b'] + p['c'] * mean_piston_speed) * mean_piston_speed
    C = numpy.power(n_temperatures, 0) * (p['l'] + p['l2'] * mean_piston_speed ** 2)
    C -= n_powers

    return A, B, C


def calculate_VMEP(A, B, C):
    if (B ** 2 - 4 * A * C) < 0:
        print('Negative (B ** 2 - 4 * A * C) ')
    if A != 0 and (B ** 2 - 4 * A * C) >= 0:
        res = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    else:
        res = float(-C) / B
    return res


def calc_fuel_consumption(VMEP, engine_capacity, engine_fuel_lower_heating_value, engine_rpm, sim_step):
    # fuel consumption in grams per time step
    res = sim_step * ((VMEP * engine_capacity * 0.1 * (float(engine_rpm) / 60)) / (2 * engine_fuel_lower_heating_value))
    if res < 0:
        return 0
    return res
