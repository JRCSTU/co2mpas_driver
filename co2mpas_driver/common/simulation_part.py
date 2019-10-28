from co2mpas_driver.common import gear_functions as fg


def accMFC(s, driver_style, sdes, acc_p_curve):
    """
    Calculate the MFC free flow acceleration.

    :param s:                   speed (m/s)
    :param driver_style:        ds 0-1
    :param sdes:                desired speed (m/s)
    :param acc_p_curve:         speed acceleration curve of the gear in use
    :return:
    """
    if s / sdes > 0.5:
        if sdes > s:
            on_off = (1 - pow(s / sdes, 60))
        else:
            on_off = 10 * (1 - s / sdes)
    else:
        on_off = (1 - 0.8 * pow(1 - s / sdes, 60))
    acc = acc_p_curve(s) * driver_style * on_off

    return acc


def clutch_on(gear_count, acc, transmission):
    """

    If clutch is on, maximum acceleration is decreased depending on the transmission

    :param gear_count:
    :param acc:
    :param transmission:
    :return:
    """

    if gear_count > 0:

        if transmission == 'manual':
            return 0.
        else:
            return acc * 2 / 3
    else:
        return acc


def simulation_step_function(transmission, speed, gear, gear_count, gs, Curves,
                             vdes, driver_style, sim_step):
    gear, gear_count = fg.gear_for_speed_profiles(gs, speed, gear, gear_count)
    acceleration = accMFC(speed, driver_style, vdes, Curves[gear - 1])
    acceleration = clutch_on(gear_count, acceleration, transmission)

    return speed + acceleration * sim_step, gear, gear_count
