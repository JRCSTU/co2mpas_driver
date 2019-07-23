#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import unittest
from new_MFC.core import dsp
import ddt
import schedula as sh
from new_MFC.test.utils import test_check

import yaml
import os.path as osp


@ddt.ddt
class TestCore(unittest.TestCase):
    def setUp(self):
        self.data, res_dir = {}, osp.join(osp.dirname(__file__), 'results')
        test_names = (
            'gear_4degree_curves_with_linear_gs', 'gear_curves_n_gs_from_poly',
            'get_ev_curve_main'
        )
        for name in test_names:
            with open(osp.join(res_dir, '%s.yaml' % name)) as f:
                self.data[name] = yaml.load(f, yaml.CLoader)

    @ddt.idata((
            # define_discrete_acceleration_curves()
            (
                    ['coefs_per_gear', 'speed_per_gear', 'Start', 'Stop',
                     'Alimit',
                     'type_of_car', 'car_type', 'veh_mass', 'engine_max_power',
                     'car_width', 'car_height', 'sp_bins'],
                    ['discrete_acceleration_curves']
            ),
            # calculate_full_load_torques()
            (
                    ['full_load_speeds', 'full_load_powers'],
                    ['full_load_torques']
            ),
            # get_speeds_n_accelerations_per_gear()
            (
                    ['gear_box_ratios', 'idle_engine_speed', 'tire_radius',
                     'driveline_slippage', 'final_drive',
                     'driveline_efficiency',
                     'veh_mass', 'full_load_speeds', 'full_load_torques'],
                    ['speed_per_gear', 'acc_per_gear']
            ),
            # gear_linear()
            (
                    ['speed_per_gear', 'gs_style'],
                    ['gs']
            ),
            # calculate_full_load_speeds_and_powers()
            (
                    ['ignition_type', 'engine_max_speed_at_max_power',
                     'engine_max_power', 'idle_engine_speed'],
                    ['full_load_speeds', 'full_load_powers']
            ),
            # get_tan_coefs()
            (
                    ['speed_per_gear', 'acc_per_gear', 'degree'],
                    ['coefs_per_gear']
            )
    ))
    def test_gear_4degree_curves_with_linear_gs(self, keys):
        (inp, out), data = keys, self.data['gear_4degree_curves_with_linear_gs']
        test_check(dsp, sh.selector(inp, data), sh.selector(out, data))

    @ddt.idata((
            # define_discrete_acceleration_curves()
            (
                    ['speed_per_gear', 'coefs_per_gear', 'use_cubic', 'Start',
                     'Stop', 'type_of_car', 'car_type', 'veh_mass',
                     'engine_max_power', 'car_width', 'car_height'],
                    ['discrete_acceleration_curves']
            ),
            # gear_points_from_tan()
            (
                    ['coefs_per_gear', 'gs_style', 'Start', 'Stop',
                     'use_linear_gs'],
                    ['gs']
            ),
            # get_resistances()
            (
                    ['type_of_car', 'car_type', 'veh_mass', 'engine_max_power',
                     'car_width', 'car_height', 'sp_bins'],
                    ['Alimit']
            ),
            #  get_start_stop()
            (
                    ['gear_box_ratios', 'veh_max_speed', 'speed_per_gear',
                     'acc_per_gear', 'use_cubic', 'coefs_per_gear'],
                    ['Start', 'Stop']
            ),
            # get_tan_coefs()
            (
                    ['speed_per_gear', 'acc_per_gear', 'degree'],
                    ['coefs_per_gear']
            ),
            # calculate_full_load_speeds_and_powers()
            (
                    ['engine_max_power', 'engine_max_speed_at_max_power',
                     'ignition_type', 'idle_engine_speed'],
                    ['full_load_powers', 'full_load_speeds']
            ),
            # calculate_full_load_torques()
            (
                    ['full_load_speeds', 'full_load_powers'],
                    ['full_load_torques']
            ),
            # get_speeds_n_accelerations_per_gear()
            (
                    ['gear_box_ratios', 'idle_engine_speed', 'tire_radius',
                     'driveline_slippage', 'final_drive', 'driveline_efficiency',
                     'veh_mass', 'full_load_speeds', 'full_load_torques'],
                    ['speed_per_gear', 'acc_per_gear']
            ),
    ))
    def test_gear_curves_n_gs_from_poly(self, keys):
        (inp, out), data = keys, self.data['gear_curves_n_gs_from_poly']
        test_check(dsp, sh.selector(inp, data), sh.selector(out, data))

    @ddt.idata((
            # ev_curve()
            (
                    ['engine_max_power', 'tire_radius', 'driveline_slippage',
                     'motor_max_torque', 'final_drive', 'gear_box_ratios',
                     'driveline_efficiency', 'veh_mass', 'veh_max_speed'],
                    ['Start', 'Stop']
            ),
    ))
    def test_get_ev_curve_main(self, keys):
        (inp, out), data = keys, self.data['get_ev_curve_main']
        test_check(dsp, sh.selector(inp, data), sh.selector(out, data))
