# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions to process a CO2MPAS input file.
"""


def gear_for_speed_profiles(gs, curr_speed, current_gear, gear_cnt,
                            clutch_duration=5):
    """
    Return the gear that must be used and the clutch condition.

    :param gs:
        Gear limits.
    :type gs: list

    :param curr_speed:
        Current speed.
    :type curr_speed: int

    :param current_gear:
        Current speed.
    :type current_gear: int

    :param gear_cnt:
        Gear count.
    :type gear_cnt: int

    :param clutch_duration:
        Clutch duration in sim step
    :type clutch_duration: int

    :return:
        Current gear & gear count
    :rtype: int, int
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
        while iter == 1 and gear_search < len(gear_limits):
            if gear_limits[gear_search - 1] <= curr_speed < gear_limits[
                gear_search]:
                gear_cnt = clutch_duration  # in simulation steps for 0.5 second
                current_gear = gear_search
                iter = 0
            else:
                gear_search += 1
        return current_gear, gear_cnt


def accMFC(velocity, driver_style, desired_velocity, acc_p_curve):
    """
    Calculate the MFC free flow acceleration.

    :param velocity:
        speed. (m/s)
    :type velocity: int

    :param driver_style:
        Driver style from 0-1.
    :type driver_style: int

    :param desired_velocity:
        desired velocity (m/s)
    :type desired_velocity: int

    :param acc_p_curve:
        Speed acceleration curve of the gear in use.
    :type acc_p_curve:

    :return:
    """
    if velocity / desired_velocity > 0.5:
        if desired_velocity > velocity:
            on_off = (1 - pow(velocity / desired_velocity, 60))
        else:
            on_off = 10 * (1 - velocity / desired_velocity)
    else:
        on_off = (1 - 0.8 * pow(1 - velocity / desired_velocity, 60))
    acc = acc_p_curve(velocity) * driver_style * on_off

    return acc


def correct_acc_clutch_on(gear_count, acc, transmission):
    """
    Get the acceleration If clutch is on. Maximum acceleration is
    decreased depending on the transmission.

    :param gear_count:
        Gear count.
    :type gear_count: int

    :param acc:
        Acceleration. (m/s2)
    :type acc:

    :param transmission:
        Transmission type.
    :type transmission: str

    :return:
        Acceleration when clutch on. (m/s2)
    :rtype:
    """

    if gear_count > 0:

        if transmission == 'manual':
            return 0.
        else:
            return acc * 2 / 3
    else:
        return acc
