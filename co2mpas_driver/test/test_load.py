#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import unittest
from co2mpas_driver.load import dsp
import ddt
from co2mpas_driver.test.utils import _check

import yaml
import os.path as osp


@ddt.ddt
class TestLoad(unittest.TestCase):
    def setUp(self):
        res_dir = osp.join(osp.dirname(__file__), 'results')
        with open(osp.join(res_dir, 'load_vehicle_data.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    @ddt.idata((
            # `merge_data`.
            ['data'],
            # `get_vehicle_inputs`.
            ['vehicle_inputs'],
            # `load_vehicle_db`.
            ['vehicle_db'],
            # `get_db_path`.
            ['db_path'],
            # `get_vehicle_id`.
            ['vehicle_id'],
            # `read_excel`.
            ['raw_data'],
    ))
    def test_load(self, out):
        _check(dsp, self.data, out)

    def test_load_v1(self):
        data = dict(
            vehicle_id=39393,
            inputs=dict(inputs=dict(duration=100, times=[1, 2, 3])), data={})
        _check(dsp, data, ['data'])
