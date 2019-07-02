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


@ddt.ddt
class Core(unittest.TestCase):
    def setUp(self):
        self.data = {
            'ignition_type': 'positive',
            'engine_max_speed_at_max_power': 5500,
            'engine_max_power': 59,
            'idle_engine_speed': (750, 50),
            'veh_mass': 915,
            'tire_radius': 0.31115000000000004,
            'driveline_efficiency': 0.9,
            'driveline_slippage': 0,
            'final_drive': 3.87,
            'gr': [4.1, 2.16, 1.34, 0.97, 0.77],
            'degree': 4,
            'gs_style': 0.8,
            'full_load_torque': [75.12113313937459, 91.17497569046989,
                                 100.46366277962001, 106.23284625088814,
                                 109.90097920550515, 112.01476232976614,
                                 113.20026887274341, 113.48089196818323,
                                 111.66015441357705, 107.96908148811838,
                                 102.014615309319, 89.89606202045873,
                                 74.24763159124234],
            'full_load_speeds': [750, 1225, 1700, 2175, 2650, 3125, 3600, 4075,
                                 4550, 5025, 5500, 5975, 6450],
            'speed_per_gear': ([2.51558902, 3.4910215, 4.46645397,
                                5.44188645, 6.41731893, 7.3927514,
                                8.36818388, 9.34361636, 10.31904883,
                                11.29448131, 12.26991379, 13.24534626],
                               [4.77496064, 6.62647599, 8.47799134,
                                10.32950669, 12.18102203, 14.03253738,
                                15.88405273, 17.73556808, 19.58708343,
                                21.43859878, 23.29011413, 25.14162948],
                               [7.69695148, 10.68148368, 13.66601589,
                                16.65054809, 19.63508029, 22.6196125,
                                25.6041447, 28.58867691, 31.57320911,
                                34.55774132, 37.54227352, 40.52680573],
                               [10.63290204, 14.75586405, 18.87882607,
                                23.00178808, 27.1247501, 31.24771211,
                                35.37067413, 39.49363614, 43.61659816,
                                47.73956017, 51.86252219, 55.9854842],
                               [13.39469478, 18.58855602, 23.78241725,
                                28.97627849, 34.17013973, 39.36400097,
                                44.55786221, 49.75172345, 54.94558469,
                                60.13944593, 65.33330717, 70.52716841]),
            'acc_per_gear': [[4.57321994, 5.03912858, 5.32850343, 5.51249227,
                              5.61851693, 5.67798041, 5.69205611, 5.60073025,
                              5.41559076, 5.11692237, 4.50907126, 3.72416605],
                             [2.40930611, 2.65476042, 2.80721156, 2.90414227,
                              2.95999916, 2.99132627, 2.99874176, 2.95062862,
                              2.85309172, 2.69574447, 2.37551072, 1.96199967],
                             [1.49466213, 1.64693471, 1.74151088, 1.80164382,
                              1.83629578, 1.85573018, 1.86033053, 1.83048257,
                              1.76997357, 1.67235999, 1.47369646, 1.21716646],
                             [1.08195691, 1.19218408, 1.26064593, 1.304175,
                              1.32925888, 1.34332707, 1.34665718, 1.32505081,
                              1.28124952, 1.21058895, 1.06678027, 0.88108319],
                             [0.85887301, 0.94637293, 1.00071894, 1.03527294,
                              1.05518489, 1.06635242, 1.0689959, 1.05184446,
                              1.01707436, 0.96098298, 0.84682558, 0.69941655]],
            'coefs_per_gear': [[-7.14516004e-04, 1.98266120e-02,
                                -2.35583660e-01, 1.41467637e+00,
                                2.22420452e+00],
                               [-2.89975585e-05, 1.52731313e-03,
                                -3.44472731e-02, 3.92642123e-01,
                                1.17177604e+00],
                               [-2.66450579e-06, 2.26220665e-04,
                                -8.22447130e-03, 1.51112011e-01,
                                7.26935135e-01],
                               [-5.29604112e-07, 6.21155046e-05,
                                -3.11967101e-03, 7.91831648e-02,
                                5.26214239e-01],
                               [-1.66934578e-07, 2.46647082e-05,
                                -1.56050718e-03, 4.98965867e-02,
                                4.17716458e-01]],
            'gs': [11.099394813678138, 21.068295711148313, 31.573209113839276,
                   43.61659815726251],
            'stating_speed': 2.5155890188262195,
            'spline_from_poly': None,
            'veh_max_speed': 48,
            'cs_acc_per_gear': None,
            'car_type': 'hatchback',
            'car_width': 1.627,
            'car_height': 1.488,
            'sp_bins': None,
            'Start': None,
            'Stop': None,
            'Alimit': None,
            'car_res_curve': None,
            'res': None

        }

    @ddt.data(
        (['ignition_type', 'engine_max_speed_at_max_power', 'engine_max_power',
          'idle_engine_speed'],
         ['full_load_torque', 'full_load_speeds']
         ),
        (
            ['veh_mass', 'tire_radius', 'full_load_speeds',
             'driveline_efficiency', 'driveline_slippage', 'final_drive',
             'full_load_torque', 'gr'],
            ['speed_per_gear', 'acc_per_gear']
        ),
        (
            ['speed_per_gear', 'acc_per_gear', 'degree'],
            ['coefs_per_gear']
        ),
        (
            ['speed_per_gear', 'gs_style'],
            ['gs']
        )

    )
    def test_core(self, keys):
        inputs, outputs = keys
        res = dsp(sh.selector(inputs, self.data), outputs)
        self.assertTrue(set(outputs).issubset(res), "Output is not a subset of result!")
        for k, v in sh.selector(outputs, self.data).items():
            if isinstance(v, str):
                self.assertEqual(v, res[k])
            else:
                nt.assert_almost_equal(res[k], v)
