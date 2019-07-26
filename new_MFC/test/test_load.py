#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import unittest
from new_MFC.load import dsp
import ddt
import schedula as sh
from new_MFC.test.utils import test_check

import yaml
import os.path as osp


@ddt.ddt
class TestLoad(unittest.TestCase):
    def setUp(self):
        with open(
                osp.join(osp.dirname(__file__), 'results', 'load_vehicle_data.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    @ddt.idata((
            # # merge_data()
            # (
            #         ['vehicle_inputs', 'raw_data', 'inputs'],
            #         ['data']
            # ),
            # get_vehicle_inputs()
            (
                    ['vehicle_id', 'vehicle_db'],
                    ['vehicle_inputs']
            ),
            # load_vehicle_db()
            (
                    ['db_path'],
                    ['vehicle_db']
            ),
            # get_db_path()
            (
                    ['raw_data'],
                    ['db_path']
            ),
            # get_vehicle_id()
            (
                    ['raw_data'],
                    ['vehicle_id']
            ),
            # read_excel()
            (
                    ['input_path'],
                    ['raw_data']
            ),
    ))
    def test_load(self, keys):
        inputs, outputs = keys
        test_check(
            dsp, sh.selector(inputs, self.data), sh.selector(outputs, self.data)
        )
