import gear_functions as fg
import numpy as np

def accMFC(s, driver_style, sdes, acc_p_curve,sim_step):
    '''

    Return the MFC free flow acceleration

    :param s:                   speed (m/s)
    :param driver_style:        ds 0-1
    :param sdes:                desired speed (m/s)
    :param acc_p_curve:         speed acceleration curve of the gear in use
    :param sim_step:            simulation step time (in seconds)
    :return:
    '''

    max_acceleration = (sdes - s)/sim_step

    r = s / sdes
    if r >= 1:
        return acc_p_curve(s) * driver_style * 10 * (1 - r)  # Deceleration.
    if r > 0.5:
        return min(acc_p_curve(s) * driver_style *(1 - np.power(r, 60)),max_acceleration)  # Acceleration.
    return acc_p_curve(s) * driver_style *(1 - 0.8 * np.power(1 - r, 60))  # Acceleration boost.

    # return acc


def clutch_on(gear_count, acc, my_car):
    '''

    If clutch is on, maximum acceleration is decreased depending on the transmission

    :param gear_count:
    :param acc:
    :param my_car:
    :return:
    '''

    if gear_count > 0:

        if my_car.transmission == 'manual':
            return 0.
        else:
            return acc * 2 / 3
    else:
        return acc


def simulation_step_function(selected_car, speed, gear, gear_count, gs, Curves, vdes, driver_style, sim_step):
    gear, gear_count = fg.gear_for_speed_profiles(gs, speed, gear, gear_count)
    acceleration = accMFC(speed, driver_style, vdes, Curves[gear - 1],sim_step)
    acceleration = clutch_on(gear_count, acceleration, selected_car)

    return speed + acceleration * sim_step, gear, gear_count
