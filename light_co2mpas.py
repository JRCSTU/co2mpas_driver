'''The two functions of light co2mpass to be used'''

import math
import numpy as np
import functions as func
import vehicle_characteristic_class as vcc
import find_gear as fg
import vehicle_functions as vf

def light_co2mpas_series(my_car, sp, gs, sim_step):
    '''

    :param my_car:
    :param sp:          In km/h!!!
    :param gs:
    :param sim_step:    in sec
    :return:
    '''

    hardcoded_params = vcc.hardcoded_params()

    n_wheel_drive = my_car.car_type
    road_loads = vf.estimate_f_coefficients(my_car, passengers=0)

    slope = 0
    # FIX First convert km/h to m/s in order to have acceleration in m/s^2
    ap = np.diff([i / (3.6 * sim_step) for i in sp])

    # gear number and gear count for shifting duration
    simulated_gear = [0, 30]
    fp = []

    if my_car.gearbox_type == 'manual':
        my_car.veh_params = hardcoded_params.params_gearbox_losses['Manual']
        my_car.gb_type = 0
    else:
        my_car.veh_params = hardcoded_params.params_gearbox_losses['Automatic']
        my_car.gb_type = 1

    ###initialize gear and gear count
    gear = 0
    gear_count = 30

    for i in range(1, len(sp)):
        speed = sp[i]
        acceleration = ap[i - 1]
        gear, gear_count = fg.gear_for_speed_profiles(gs, speed / 3.6, gear, gear_count, my_car.gb_type)
        fc = light_co2mpas_instant(my_car, speed, acceleration, hardcoded_params, road_loads, slope, gs, gear,
                                   gear_count,
                                   sim_step)

        fp.append(fc)

    return fp


def light_co2mpas_instant(my_car, speed, acceleration, hardcoded_params, road_loads, slope, gs, gear, gear_count,
                          sim_step):
    n_wheel_drive = my_car.car_type

    # The power on wheels in kW
    # Check if it is current or previous speed
    veh_wheel_power = \
        func.calculate_wheel_power \
            (speed, acceleration, road_loads,
             my_car.veh_mass, slope)

    # The speed on the wheels in [RPM]
    veh_wheel_speed = \
        func.calculate_wheel_speeds \
            (speed, my_car.r_dynamic)

    # # The torque on the wheels in [N*m]
    veh_wheel_torque = \
        func.calculate_wheel_torques \
            (veh_wheel_power, veh_wheel_speed)

    # Calculates final drive speed in RPM
    final_drive_speed = \
        func.calculate_final_drive_speeds_in \
            (veh_wheel_speed, my_car.final_drive)

    # Final drive torque losses [N*m].
    final_drive_torque_losses = \
        func.calculate_final_drive_torque_losses_v1 \
            (n_wheel_drive, \
             veh_wheel_torque, my_car.final_drive, \
             hardcoded_params.final_drive_efficiency)

    # Final drive torque in [N*m].
    final_drive_torque_in = \
        func.calculate_final_drive_torques_in \
            (veh_wheel_torque, my_car.final_drive, \
             final_drive_torque_losses)

    # FIX Input of speeds in m/s
    # gear, gear_count = acc.gear_for_speed_profiles(gs, speed / 3.6, gear,gear_count, my_car.gb_type)

    gear_box_speeds_in = \
        func.calculate_gear_box_speeds_in_v1 \
            (gear, final_drive_speed,
             my_car.gr, 0)

    gearbox_params = \
        func.create_gearbox_params \
            (my_car.veh_params, my_car.engine_max_torque)

    gear_box_torques_in = \
        func.gear_box_torques_in \
            (hardcoded_params.min_engine_on_speed, final_drive_torque_in,
             gear_box_speeds_in,
             final_drive_speed, gearbox_params, gear_count)

    gear_box_power_out = \
        2 * math.pi * gear_box_torques_in * gear_box_speeds_in / 60000

    br_eff_pres = \
        func.calculate_brake_mean_effective_pressures \
            (gear_box_speeds_in, gear_box_power_out,
             my_car.fuel_eng_capacity, hardcoded_params.min_engine_on_speed)

    engine_cm = func.mean_piston_speed(gear_box_speeds_in, my_car.fuel_engine_stroke)

    params = func.parameters \
        (my_car.max_power, my_car.fuel_eng_capacity, my_car.fuel_type, my_car.fuel_turbo)
    fuel_A, fuel_B, fuel_C = func.calculate_fuel_ABC \
        (params, engine_cm, br_eff_pres, 100)

    if br_eff_pres > 20:
        print('BMEP> %.2f bar, EngineCM: %.2f, Gear: %d : Check out the MFC output. The engine will blow up!!!!' % (
            br_eff_pres, engine_cm, gear))

    if br_eff_pres > -0.5:
        VMEP = func.calculate_VMEP \
            (fuel_A, fuel_B, fuel_C)
    else:
        VMEP = 0
    lower_heating_value = hardcoded_params.LHV[my_car.fuel_type]

    # fuel consumption in grams per time step
    fc = func.calc_fuel_consumption(VMEP, my_car.fuel_eng_capacity, lower_heating_value, gear_box_speeds_in, sim_step)

    return fc

##   OLDER FUNCTION
# def light_co2mpas_series(my_car, sp, gs, sim_step):
#
#     ## speed is in km/h
#
#     hardcoded_params = vcc.hardcoded_params()
#
#     n_wheel_drive = my_car.car_type
#     road_loads = func.estimate_f_coefficients(my_car.type_of_car, my_car.car_width, my_car.car_height,
#                                               my_car.kerb_weight, passengers=0)
#
#     slope = 0
#     # FIX First convert km/h to m/s in order to have acceleration in m/s^2
#     ap = np.diff([i / (3.6 * sim_step) for i in sp])
#
#     # gear number and gear count for shifting duration
#     simulated_gear = [0, 30]
#     # tmp_find_gear = [0, 30]
#     fp = []
#     lgear_box_speeds_in, lgear_box_torques_in = [], []
#     for i in range(1, len(sp)):
#         # The power on wheels in kW
#         # Check if it is current or previous speed
#         veh_wheel_power = \
#             func.calculate_wheel_power \
#                 (sp[i], ap[i - 1], road_loads,
#                  my_car.veh_mass, slope)
#
#         # The speed on the wheels in [RPM]
#         veh_wheel_speed = \
#             func.calculate_wheel_speeds \
#                 (sp[i], my_car.r_dynamic)
#
#         # # The torque on the wheels in [N*m]
#         veh_wheel_torque = \
#             func.calculate_wheel_torques \
#                 (veh_wheel_power, veh_wheel_speed)
#
#         # Calculates final drive speed in RPM
#         final_drive_speed = \
#             func.calculate_final_drive_speeds_in \
#                 (veh_wheel_speed, my_car.final_drive)
#
#         # Final drive torque losses [N*m].
#         final_drive_torque_losses = \
#             func.calculate_final_drive_torque_losses_v1 \
#                 (n_wheel_drive, \
#                  veh_wheel_torque, my_car.final_drive, \
#                  hardcoded_params.final_drive_efficiency)
#
#         # Final drive torque in [N*m].
#         final_drive_torque_in = \
#             func.calculate_final_drive_torques_in \
#                 (veh_wheel_torque, my_car.final_drive, \
#                  final_drive_torque_losses)
#
#         if my_car.gearbox_type == 'manual':
#             veh_params = hardcoded_params.params_gearbox_losses['Manual']
#             gb_type = 0
#         else:
#             veh_params = hardcoded_params.params_gearbox_losses['Automatic']
#             gb_type = 1
#
#         # FIX Input of speeds in m/s
#         simulated_gear[0], simulated_gear[1] = acc.gear_for_speed_profiles(gs, sp[i] / 3.6, simulated_gear[0],
#                                                                                simulated_gear[1], gb_type)
#
#
#         gear_box_speeds_in = \
#             func.calculate_gear_box_speeds_in_v1 \
#                 (simulated_gear[0], final_drive_speed,
#                  my_car.gr, 0)
#         lgear_box_speeds_in.append(gear_box_speeds_in)
#
#         gearbox_params = \
#             func.create_gearbox_params \
#                 (veh_params, my_car.engine_max_torque)
#
#         gear_box_torques_in = \
#             func.gear_box_torques_in \
#                 (hardcoded_params.min_engine_on_speed, final_drive_torque_in,
#                  gear_box_speeds_in,
#                  final_drive_speed, gearbox_params, simulated_gear[1])
#         lgear_box_torques_in.append(gear_box_torques_in)
#
#         gear_box_power_out = \
#             2 * math.pi * gear_box_torques_in * gear_box_speeds_in / 60000
#
#         br_eff_pres = \
#             func.calculate_brake_mean_effective_pressures \
#                 (gear_box_speeds_in, gear_box_power_out,
#                  my_car.fuel_eng_capacity, hardcoded_params.min_engine_on_speed)
#
#         engine_cm = func.mean_piston_speed(gear_box_speeds_in, my_car.fuel_engine_stroke)
#
#         params = func.parameters \
#             (my_car.max_power, my_car.fuel_eng_capacity, my_car.fuel_type, my_car.fuel_turbo)
#         fuel_A, fuel_B, fuel_C = func.calculate_fuel_ABC \
#             (params, engine_cm, br_eff_pres, 100)
#
#         if br_eff_pres > 20:
#             print('BMEP> %.2f bar, EngineCM: %.2f, Gear: %d : Check out the MFC output. The engine will blow up!!!!' % (
#                 br_eff_pres, engine_cm, simulated_gear[0]))
#
#         if br_eff_pres > -0.5:
#             VMEP = func.calculate_VMEP \
#                 (fuel_A, fuel_B, fuel_C)
#         else:
#             VMEP = 0
#         lower_heating_value = hardcoded_params.LHV[my_car.fuel_type]
#
#         # fuel consumption in grams per time step
#         fc = func.calc_fuel_consumption \
#             (VMEP, my_car.fuel_eng_capacity, lower_heating_value, \
#              gear_box_speeds_in, sim_step)
#         fp.append(fc)
#
#     return fp
