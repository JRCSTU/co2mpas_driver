#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import sys
sys.path.append("..")
import unittest
from stu_mfc import dsp
import ddt
import schedula as sh
import numpy.testing as nt

import yaml
import os.path as osp


@ddt.ddt
class Core(unittest.TestCase):
    def setUp(self):
        with open(osp.join(osp.dirname(__file__), 'res.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    @ddt.idata((
        (
             ['gearbox_type', 'veh_params', 'gb_type', 'car_type',
              'veh_mass', 'r_dynamic', 'final_drive', 'gr',
              'engine_max_torque', 'max_power', 'fuel_eng_capacity',
              'fuel_engine_stroke', 'fuel_type', 'fuel_turbo', 'type_of_car',
              'car_width', 'car_height', 'sp', 'gs', 'sim_step'],
             ['fp']
         ),
        (
             ['gs', 'curr_speed', 'current_gear', 'gear_cnt',
              'clutch_duration'],
             ['current_gear', 'gear_cnt']
        ),
        (
            ['type_of_car', 'car_type', 'veh_mass', 'engine_max_power',
             'car_width', 'car_height', 'sp_bins'],
            ['Alimit']
        ),
        (
            ['gr', 'veh_max_speed', 'speed_per_gear', 'acc_per_gear',
             'coefs_per_gear', 'starting_speed'],
            ['Start', 'Stop']
        ),
        (
            ['ignition_type', 'engine_max_speed_at_max_power',
             'engine_max_power', 'idle_engine_speed'],
            ['full_load_torque', 'full_load_speeds']
         ),
        (
            ['veh_mass', 'tire_radius', 'full_load_speeds',
             'driveline_efficiency', 'driveline_slippage', 'final_drive',
             'full_load_torque', 'gr', 'idle_engine_speed'],
            ['speed_per_gear', 'acc_per_gear']
        ),
        (
            ['speed_per_gear', 'acc_per_gear', 'degree'],
            ['coefs_per_gear']
        ),
        (
            ['speed_per_gear', 'gs_style'],
            ['gs']
        ),
    ))
    def test_core(self, keys):
        inputs, outputs = keys
        res = dsp(sh.selector(inputs, self.data), outputs)
        self.assertTrue(set(outputs).issubset(res), "Missing outputs {}".format(set(outputs) - set(res)))
        for k, v in sh.selector(outputs, self.data).items():
            if isinstance(v, str):
                self.assertEqual(v, res[k])
            else:
                nt.assert_almost_equal(res[k], v)
