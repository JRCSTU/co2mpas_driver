#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
import unittest
from co2mpas_driver import dsp
import ddt
from tests.utils import _check
import yaml
import os.path as osp
import schedula as sh


@ddt.ddt
class TestCore(unittest.TestCase):
    def setUp(self):
        res_dir = osp.join(osp.dirname(__file__), "results")
        out_fpath = "gear_4degree_curves_with_linear_gs_core.yaml"
        with open(osp.join(res_dir, out_fpath)) as f:
            self.data = dict(outputs=yaml.load(f, yaml.CLoader))
        with open(osp.join(res_dir, "core.yaml")) as f:
            sh.combine_nested_dicts(yaml.load(f, yaml.CLoader), base=self.data, depth=2)

    def test_sample_simulation(self):
        _check(dsp, self.data, ["outputs"])
