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
from new_MFC.test.utils import _assert
import yaml
import os.path as osp


@ddt.ddt
class Core(unittest.TestCase):
    def setUp(self):
        with open(osp.join(osp.dirname(__file__),
                           'gear_4degree_curves_with_linear_gs.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    @ddt.idata((
        # define_discrete_acceleration_curves()
        (
                ['coefs_per_gear', 'speed_per_gear', 'Start', 'Stop', 'Alimit',
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
                 'driveline_slippage', 'final_drive', 'driveline_efficiency',
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
    def test_core(self, keys):
        inputs, outputs = keys
        res = dsp(sh.selector(inputs, self.data), outputs)
        self.assertTrue(set(outputs).issubset(res), "Missing outputs {}".format(set(outputs) - set(res)))
        for k, v in sh.selector(outputs, self.data).items():
            _assert(v, res[k])
