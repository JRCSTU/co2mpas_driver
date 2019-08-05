# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to processes a CO2MPAS input file.
"""


def gear_for_speed_profiles(gs, curr_speed, current_gear, gear_cnt,
                            clutch_duration=5):
    """
    Return the gear that must be used and the clutch condition

    :param gs:
    :param curr_speed:
    :param current_gear:
    :param gear_cnt:
    :param clutch_duration: in sim step
    :return:
    """

    # Model buffer for up shifting and down shifting.
    upshift_offs = 0.0
    downshift_off = 0.1

    gear_limits = [0]
    gear_limits.extend(gs)
    gear_limits.append(200)

    if gear_limits[current_gear - 1] - gear_limits[
        current_gear - 1] * downshift_off <= curr_speed < gear_limits[
        current_gear] + gear_limits[current_gear] * upshift_offs:
        if gear_cnt == 0:
            return current_gear, gear_cnt
        else:
            gear_cnt -= 1
            return current_gear, gear_cnt
    else:
        iter = 1
        gear_search = 1
        while iter == 1:
            if gear_limits[gear_search - 1] <= curr_speed < gear_limits[
                gear_search]:
                gear_cnt = clutch_duration  # in simulation steps for 0.5 second
                current_gear = gear_search
                iter = 0
            else:
                gear_search += 1
        return current_gear, gear_cnt


def accMFC(s, driver_style, sdes, acc_p_curve):
    """

    Return the MFC free flow acceleration

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


def simulation_step_function(
        transmission, speed, gear, gear_count, gs, Curves,
        vdes, driver_style, sim_step):
    gear, gear_count = gear_for_speed_profiles(gs, speed, gear, gear_count)
    acceleration = accMFC(speed, driver_style, vdes, Curves[gear - 1])
    acceleration = clutch_on(gear_count, acceleration, transmission)

    return speed + acceleration * sim_step, gear, gear_count
