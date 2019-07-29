#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import unittest
from new_MFC import dsp
import ddt
from new_MFC.test.utils import test_check
import numpy as np
import yaml
import os.path as osp


@ddt.ddt
class TestCore(unittest.TestCase):
    def setUp(self):
        res_dir = osp.join(osp.dirname(__file__), 'results')
        with open(osp.join(res_dir, 'load_vehicle_data.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    def test_sample_simulation(self):
        sol = dsp(dict(inputs=dict(inputs=dict(
            gs_style=0.9,
            v_des=40,
            v_start=0,
            sim_step=0.1,
            driver_style=1,
            duration=100,
            times=np.arange(0, 100 + 0.1, 0.1)
        )), vehicle_id=39393))
        test_check(dsp, self.data, out)
