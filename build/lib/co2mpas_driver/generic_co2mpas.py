"""The two functions of light co2mpass to be used"""

import numpy as np
from co2mpas_driver.common import functions as func
from co2mpas_driver.common import vehicle_specs_class as vcc
from co2mpas_driver.common import gear_functions as fg
from co2mpas_driver.common import vehicle_functions as vf


def light_co2mpas_series(vehicle_mass, r_dynamic, car_type, final_drive,
                          gear_box_ratios, gearbox_type, gb_type, veh_params,
                         type_of_car, car_width, car_height,
                         engine_max_torque, fuel_eng_capacity, max_power,
                         fuel_engine_stroke, fuel_type, fuel_turbo, sim_step,
                         sp, gs, **kwargs):
    """
    :param my_car:
    :param sp:          In km/h!!!
    :param gs:
    :param sim_step:    in sec
    :param gb_type:
    :param veh_params:
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
        clutch_list = fg.create_clutch_list(gear_list, clutch_duration)

    hardcoded_params = vcc.HardcodedParams()

    # n_wheel_drive = my_car.car_type
    road_loads = vf.estimate_f_coefficients(type_of_car, vehicle_mass,
                                            car_width, car_height, passengers=0)

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
            gear, gear_count = fg.gear_for_speed_profiles(gs, speed / 3.6, gear,
                                                          gear_count, gb_type)

        fc = light_co2mpas_instant(vehicle_mass, r_dynamic, car_type, final_drive,
                          gear_box_ratios, veh_params, engine_max_torque,
                          fuel_eng_capacity, speed, acceleration, max_power,
                          fuel_engine_stroke, fuel_type, fuel_turbo,
                          hardcoded_params, road_loads,  slope, gear,
                          gear_count, sim_step)

        fp.append(fc)

    return fp


def light_co2mpas_instant(vehicle_mass, r_dynamic, car_type, final_drive,
                          gear_box_ratios, veh_params, engine_max_torque,
                          fuel_eng_capacity, speed, acceleration, max_power,
                          fuel_engine_stroke, fuel_type, fuel_turbo,
                          hardcoded_params, road_loads,  slope, gear,
                          gear_count, sim_step):
    n_wheel_drive = car_type

    # The power on wheels in kW
    veh_wheel_power = \
        func.calculate_wheel_power \
            (speed, acceleration, road_loads,
             vehicle_mass, slope)

    # The speed on the wheels in [RPM]
    veh_wheel_speed = \
        func.calculate_wheel_speeds \
            (speed, r_dynamic)

    # # The torque on the wheels in [N*m]
    veh_wheel_torque = \
        func.calculate_wheel_torques \
            (veh_wheel_power, veh_wheel_speed)

    # Calculates final drive speed in RPM
    final_drive_speed = \
        func.calculate_final_drive_speeds_in \
            (veh_wheel_speed, final_drive)

    # Final drive torque losses [N*m].
    final_drive_torque_losses = \
        func.calculate_final_drive_torque_losses_v1 \
            (n_wheel_drive, \
             veh_wheel_torque, final_drive, \
             hardcoded_params.final_drive_efficiency)

    # Final drive torque in [N*m].
    final_drive_torque_in = \
        func.calculate_final_drive_torques_in \
            (veh_wheel_torque, final_drive, \
             final_drive_torque_losses)

    gear_box_speeds_in = \
        func.calculate_gear_box_speeds_in_v1 \
            (gear, final_drive_speed,
             gear_box_ratios, 0)

    gearbox_params = \
        func.create_gearbox_params \
            (veh_params, engine_max_torque)

    gear_box_torques_in = \
        func.gear_box_torques_in \
            (hardcoded_params.min_engine_on_speed, final_drive_torque_in,
             gear_box_speeds_in,
             final_drive_speed, gearbox_params, gear_count)

    gear_box_power_out = func.calculate_gear_box_power_out(gear_box_torques_in,
                                                           gear_box_speeds_in)

    br_eff_pres = \
        func.calculate_brake_mean_effective_pressures \
            (gear_box_speeds_in, gear_box_power_out,
             fuel_eng_capacity, hardcoded_params.min_engine_on_speed)

    engine_cm = func.mean_piston_speed(gear_box_speeds_in, fuel_engine_stroke)

    params = func.parameters \
        (max_power, fuel_eng_capacity, fuel_type, fuel_turbo)
    fuel_A, fuel_B, fuel_C = func.calculate_fuel_ABC \
        (params, engine_cm, br_eff_pres, 100)

    if br_eff_pres > 20:
        # Control for unrealistic Break Mean Effective Pressure values.
        print('BMEP> %.2f bar, EngineCM: %.2f, Gear: %d : Check out the MFC output. The engine will blow up!!!!' % (
            br_eff_pres, engine_cm, gear))

    if br_eff_pres > -0.5:
        # Fuel mean effective pressure
        VMEP = func.calculate_VMEP \
            (fuel_A, fuel_B, fuel_C)
    else:
        VMEP = 0
    lower_heating_value = hardcoded_params.LHV[fuel_type]

    # Fuel consumption in grams.
    fc = func.calc_fuel_consumption(VMEP, fuel_eng_capacity,
                                    lower_heating_value, gear_box_speeds_in,
                                    sim_step)

    return fc
