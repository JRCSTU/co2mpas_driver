import schedula as sh
import numpy as np
import co2mpas as cp

dsp = sh.Dispatcher()


# Full load curves of speed and torque
@sh.add_function(dsp, outputs=['full_load_speeds', 'full_load_torque'])
def get_load_speed_n_torque(ignition_type):
    """
    :param ignition_type:
        Engine ignition type (positive or compression).
    :type ignition_type: str
    :return:
    """
    full_load = get_full_load(ignition_type)
    full_load_speeds, full_load_powers = calculate_full_load_speeds_and_powers(full_load, my_car)
    full_load_torque = full_load_powers * 1000 * (full_load_speeds / 60 * 2 * np.pi) ** -1
    return full_load_speeds, full_load_torque


# Speed and acceleration ranges and points for each gear
@sh.add_function(dsp, outputs=['speed_per_gear', 'acc_per_gear'])
def get_speeds_n_accelerations_per_gear(gr, idle_engine_speed, tire_radius, driveline_slippage,
            final_drive, driveline_efficiency, veh_mass,
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

    for j in range(len(my_car.gr)):
        mask = full_load_speeds > 1.25 * my_car.idle_engine_speed[0]

        temp_speed = 2 * np.pi * my_car.tire_radius * full_load_speeds[mask] * (
                    1 - my_car.driveline_slippage) / (
                             60 * my_car.final_drive * my_car.gr[j])
        speed_per_gear.append(temp_speed)

        temp_acc = full_load_torque[mask] * (my_car.final_drive * my_car.gr[
            j]) * my_car.driveline_efficiency / (
                           my_car.tire_radius * my_car.veh_mass)

        acc_per_gear.append(temp_acc)

    return speed_per_gear, acc_per_gear

# Extract speed acceleration Splines
dsp.add_function(
    function_id='get_tan_coefs',
    inputs=['speed_per_gear', 'acc_per_gear', '4'],
    outputs=['coefs_per_gear']
)

# Extract speed acceleration Splines
dsp.add_function(
    function_id='get_spline_out_of_coefs',
    inputs=['coefs_per_gear', 'speed_per_gear'],
    outputs=['poly_spline']
)

# Start/stop speed for each gear
dsp.add_function(
    function_id='get_start_stop',
    inputs=['gr', 'veh_max_speed', 'speed_per_gear', 'acc_per_gear',
            'poly_spline'],
    outputs=['Start', 'Stop']
)

# Start/stop speed for each gear
dsp.add_function(
    function_id='arange',
    inputs=['0', 'Stop[-1] + 1', '0.01'],
    outputs=['sp_bins']
)

# Calculate Curves
dsp.add_function(
    function_id='get_resistances',
    inputs=['car_type', 'veh_mass', 'engine_max_power', 'type_of_car',
            'car_width', 'car_height', 'sp_bins'],
    outputs=['car_res_curve', 'car_res_curve_force', 'Alimit']
)

# Extract speed acceleration Splines
dsp.add_function(
    function_id='gear_linear',
    inputs=['speed_per_gear', 'gs_style'],
    outputs=['gs']
)

dsp.add_function(
    function_id='calculate_curves_to_use',
    inputs=['poly_spline', 'Start', 'Stop', 'Alimit', 'car_res_curve',
            'sp_bins'],
    outputs=['Curves']
)

if __name__ == '__main__':
    dsp.plot()
