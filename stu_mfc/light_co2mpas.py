import schedula as sh
from stu_mfc.functions import *


def light_co2mpas():
    """
    Defines the wheels model.

    .. dispatcher:: d

        >>> dsp = light_co2mpas()

    :return:
        Fuel consumption.
    :rtype: schedula.Dispatcher
    """

    dsp = sh.Dispatcher(
        name='Light co2mpas instant',
        description='It calculates fuel consumption.'
    )
    dsp.add_function(
        function=calculate_wheel_power,
        inputs=['speed', 'acceleration', 'road_loads', 'veh_mass', 'slope'],
        outputs=['veh_wheel_power']
    )
    dsp.add_func(
        function=calculate_wheel_speeds,
        inputs=['speed', 'r_dynamic'],
        outputs=['veh_wheel_speed']
    )
    dsp.add_func(
        function=calculate_wheel_torques,
        inputs=['veh_wheel_power', 'veh_wheel_speed'],
        outputs=['veh_wheel_torque']
    )
    dsp.add_func(
        function=calculate_final_drive_speeds_in,
        inputs=['veh_wheel_speed', 'final_drive'],
        outputs=['final_drive_speed']
    )
    dsp.add_func(
        calculate_final_drive_torque_losses_v1,
        inputs=['n_wheel_drive', 'veh_wheel_torque', 'final_drive', 'final_drive_efficiency'],
        outputs=['final_drive_torque_losses']
    )
    dsp.add_func(
        function=calculate_final_drive_torques_in,
        inputs=['veh_wheel_torque, final_drive', 'final_drive_torque_losses'],
        outputs=['final_drive_torque_in']
    )
    dsp.add_func(
        calculate_gear_box_speeds_in_v1,
        inputs=['gear', 'final_drive_speed', 'gr', 'clutch'],
        outputs=['gear_box_speeds_in']
    )
    dsp.add_func(
        function=create_gearbox_params,
        inputs=['veh_params', 'engine_max_torque'],
        outputs=['gearbox_params']
    )
    dsp.add_func(
        function=gear_box_torques_in,
        inputs=['min_engine_on_speed', 'final_drive_torque_in', 'gear_box_speeds_in', 'final_drive_speed',
                'gearbox_params', 'gear_count'],
        outputs=['gear_box_torques_in_']
    )
    dsp.add_func(
        function=calculate_gear_box_power_out,
        inputs=['gear_box_torques_in', 'gear_box_speeds_in'],
        outputs=['gear_box_power_out']
    )
    dsp.add_func(
        function=calculate_brake_mean_effective_pressures,
        inputs=['gear_box_speeds_in', 'gear_box_power_out', 'fuel_eng_capacity', 'min_engine_on_speed'],
        outputs=['br_eff_pres']
    )
    dsp.add_func(
        function=mean_piston_speed,
        inputs=['gear_box_speeds_in', 'fuel_engine_stroke'],
        outputs=['engine_cm']
    )
    dsp.add_func(
        function=parameters,
        inputs=['max_power', 'fuel_eng_capacity', 'fuel_type', 'fuel_turbo'],
        outputs=['params']
    )
    dsp.add_func(
        function=calculate_fuel_ABC,
        inputs=['params', 'engine_cm', 'br_eff_pres', 100],
        outputs=['fuel_A', 'fuel_B', 'fuel_C']
    )
    dsp.add_func(
        function=calculate_VMEP,
        inputs=['fuel_A', 'fuel_B', 'fuel_C'],
        outputs=['VMEP']
    )
    dsp.add_func(
        function=calc_fuel_consumption,
        inputs=['VMEP', 'fuel_eng_capacity', 'lower_heating_value', 'gear_box_speeds_in', 'sim_step'],
        outputs=['fc']
    )

    return dsp