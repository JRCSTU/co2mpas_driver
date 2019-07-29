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
class TestCore(unittest.TestCase):
    def setUp(self):
        res_dir = osp.join(osp.dirname(__file__), 'results')
        with open(osp.join(res_dir, 'core.yaml')) as f:
            self.data = yaml.load(f, yaml.CLoader)

    def test_sample_simulation(self):
        _check(dsp, self.data, ['outputs'])
