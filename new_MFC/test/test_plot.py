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
from new_MFC.test.utils import _check

import yaml
import os.path as osp


@ddt.ddt
class TestPlot(unittest.TestCase):
    def setUp(self):
        res_dir = osp.join(osp.dirname(__file__), 'results')
        with open(osp.join(res_dir, 'plot.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    def test_load(self, out):
        _check(dsp, self.data, out)

    def test_load_v1(self):
        data = dict(
            vehicle_id=39393,
            inputs=dict(inputs=dict(duration=100, times=[1, 2, 3])), data={})
        _check(dsp, data, ['data'])
