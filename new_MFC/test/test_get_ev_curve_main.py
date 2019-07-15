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
import numpy.testing as nt

import yaml
import os.path as osp


@ddt.ddt
class Core(unittest.TestCase):
    def setUp(self):
        with open(osp.join(osp.dirname(__file__),
                           'get_ev_curve_main.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    @ddt.idata((
            # ev_curve()
            (
                    ['engine_max_power', 'tire_radius', 'driveline_slippage',
                     'motor_max_torque', 'final_drive', 'gear_box_ratios',
                     'driveline_efficiency', 'veh_mass', 'veh_max_speed'],
                    ['Start', 'Stop']
            ),
            # get_resistances()
            (
                    ['type_of_car', 'car_type', 'veh_mass', 'engine_max_power',
                    'car_width', 'car_height', 'sp_bins'],
                    ['Alimit']
            ),
    ))
    def test_core(self, keys):
        inputs, outputs = keys
        res = dsp(sh.selector(inputs, self.data), outputs)
        self.assertTrue(set(outputs).issubset(res),
                        "Missing outputs {}".format(set(outputs) - set(res)))
        for k, v in sh.selector(outputs, self.data).items():
            if isinstance(v, str):
                self.assertEqual(v, res[k])
            else:
                nt.assert_almost_equal(res[k], v)
