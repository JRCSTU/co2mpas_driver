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
            # `get_spline_out_of_coefs`.
            ['discrete_poly_spline'],
            # `define_discrete_car_res_curve`
            ['discrete_car_res_curve'],
            # `define_discrete_car_res_curve_force`
            ['discrete_car_res_curve_force'],
            # `get_load_speed_n_torque`.
            ['full_load_speeds', 'full_load_torques'],
            # `get_speeds_n_accelerations_per_gear`.
            ['speed_per_gear', 'acc_per_gear'],
            # `get_tan_coefs`.
            ['coefs_per_gear'],
            # `get_start_stop`.
            ['Start', 'Stop', 'sp_bins'],
            # `get_resistances`.
            ['Alimit'],
            # `define_discrete_acceleration_curves`.
            ['discrete_acceleration_curves'],
            # `gear_linear`.
            ['gs']

            ## define_discrete_acceleration_curves()
            #['discrete_acceleration_curves'],
            ## calculate_full_load_torques()
            #['full_load_torques'],
            ## get_speeds_n_accelerations_per_gear()
            #['speed_per_gear', 'acc_per_gear'],
            ## gear_linear()
            #['gs'],
            ## calculate_full_load_speeds_and_powers()
            #['full_load_speeds', 'full_load_powers'],
            ## get_tan_coefs()
            #['coefs_per_gear']
    ))
    def test_gear_4degree_curves_with_linear_gs(self, out):
        test_check(dsp, self.data['gear_4degree_curves_with_linear_gs'], out)

    @ddt.idata((
            # `get_spline_out_of_coefs`.
            ['discrete_poly_spline'],
            # `define_discrete_car_res_curve`.
            ['discrete_car_res_curve'],
            # `define_discrete_car_res_curve_force`.
            ['discrete_car_res_curve_force'],
            # `define_discrete_acceleration_curves`.
            ['discrete_acceleration_curves'],
            # `gear_points_from_tan`.
            ['gs'],
            # `get_resistances`.
            ['Alimit'],
            # `get_start_stop`.
            ['Start', 'Stop'],
            # `get_tan_coefs`.
            ['coefs_per_gear'],
            # `calculate_full_load_speeds_and_powers`.
            ['full_load_powers', 'full_load_speeds'],
            # `calculate_full_load_torques`.
            ['full_load_torques'],
            # `get_speeds_n_accelerations_per_gear`.
            ['speed_per_gear', 'acc_per_gear'],
    ))
    def test_gear_curves_n_gs_from_poly(self, out):
        test_check(dsp, self.data['gear_curves_n_gs_from_poly'], out)

    @ddt.idata((
            # `define_discrete_acceleration_curves`.
            ['discrete_acceleration_curves'],
            # `get_resistances`.
            ['Alimit'],
            # `ev_curve`.
            ['Start', 'Stop'],
    ))
    def test_get_ev_curve_main(self, out):
        test_check(dsp, self.data['get_ev_curve_main'], out)
